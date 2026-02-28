"""Continual learning dataset utilities.

Provides task sequence generators for:
1. Split-class continual classification (disjoint label subsets)
2. Template-shift distribution shift (same labels, different prompts)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TaskData:
    """A single continual learning task."""

    task_id: int
    train_features: Tensor
    train_labels: Tensor
    test_features: Tensor
    test_labels: Tensor
    classes: list[int]


class SplitDatasetSequence:
    """Splits a dataset into sequential tasks by class label.

    Example: MNIST with 5 tasks of 2 classes each:
      Task 0: digits 0,1
      Task 1: digits 2,3
      ...
    """

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
        test_features: Tensor,
        test_labels: Tensor,
        n_tasks: int = 5,
        classes_per_task: int = 2,
        seed: int = 42,
    ):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task

        # Determine class ordering
        all_classes = sorted(labels.unique().tolist())
        num_classes = len(all_classes)
        required = n_tasks * classes_per_task
        if required > num_classes:
            raise ValueError(
                f"n_tasks * classes_per_task ({n_tasks} * {classes_per_task} = {required}) "
                f"exceeds available classes ({num_classes})"
            )

        rng = random.Random(seed)
        rng.shuffle(all_classes)

        # Split into task groups
        self.task_classes: list[list[int]] = []
        for i in range(n_tasks):
            start = i * classes_per_task
            end = start + classes_per_task
            self.task_classes.append(all_classes[start:end])

    def get_task(self, task_id: int) -> TaskData:
        """Get train/test data for a single task."""
        classes = self.task_classes[task_id]
        class_set = set(classes)

        # Filter train
        train_mask = torch.tensor([l.item() in class_set for l in self.labels])
        # Filter test
        test_mask = torch.tensor([l.item() in class_set for l in self.test_labels])

        return TaskData(
            task_id=task_id,
            train_features=self.features[train_mask],
            train_labels=self.labels[train_mask],
            test_features=self.test_features[test_mask],
            test_labels=self.test_labels[test_mask],
            classes=classes,
        )

    def get_all_seen_test(self, up_to_task: int) -> tuple[Tensor, Tensor]:
        """Get test data for all classes seen up to (inclusive) given task."""
        all_classes = set()
        for t in range(up_to_task + 1):
            all_classes.update(self.task_classes[t])

        mask = torch.tensor([l.item() in all_classes for l in self.test_labels])
        return self.test_features[mask], self.test_labels[mask]


@dataclass
class PromptTemplate:
    """Template for generating text classification prompts."""

    template: str  # must contain {text} placeholder
    name: str = ""


# Default templates for template-shift benchmark
DEFAULT_TEMPLATES = [
    PromptTemplate("Label this as a category: {text}", "direct"),
    PromptTemplate("Category? {text}", "question"),
    PromptTemplate("Decide which group this belongs to: {text}", "indirect"),
    PromptTemplate("Classify the following text: {text}", "classify"),
]


class TemplateShiftSequence:
    """Same labels, different prompt templates per phase.

    Tests robustness to distribution shift in feature space
    when the base LLM is frozen.
    """

    def __init__(
        self,
        raw_texts: list[str],
        labels: list[int],
        templates: list[PromptTemplate] | None = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.raw_texts = raw_texts
        self.labels = labels
        self.templates = templates or DEFAULT_TEMPLATES
        self.n_tasks = len(self.templates)

        # Train/test split (same split for all templates)
        rng = random.Random(seed)
        indices = list(range(len(raw_texts)))
        rng.shuffle(indices)
        split = int(len(indices) * train_ratio)
        self.train_indices = indices[:split]
        self.test_indices = indices[split:]

    def get_prompts(self, task_id: int, indices: list[int]) -> list[str]:
        """Apply template to raw texts for given task/phase."""
        template = self.templates[task_id]
        return [template.template.format(text=self.raw_texts[i]) for i in indices]

    def get_train_prompts(self, task_id: int) -> tuple[list[str], list[int]]:
        prompts = self.get_prompts(task_id, self.train_indices)
        labels = [self.labels[i] for i in self.train_indices]
        return prompts, labels

    def get_test_prompts(self, task_id: int) -> tuple[list[str], list[int]]:
        prompts = self.get_prompts(task_id, self.test_indices)
        labels = [self.labels[i] for i in self.test_indices]
        return prompts, labels
