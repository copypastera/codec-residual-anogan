"""Best-config BCE AnoGAN training primitives."""
import math

import numpy as np
import torch
import torch.nn as nn


def build_optimizers(generator, discriminator, config):
    training = config["training"]
    betas = tuple(float(value) for value in training.get(
        "adam_betas", [0.5, 0.999]))
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(training["generator_learning_rate"]), betas=betas)
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training["discriminator_learning_rate"]), betas=betas)
    decay = float(training.get("learning_rate_decay", 1.0))
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=decay)
    return (optimizer_g, optimizer_d), (scheduler_g, scheduler_d)


def _gradient_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def train_epoch(generator, discriminator, features, optimizers, config,
                device, epoch, seed, status_callback=None):
    """Train one epoch using the released BCE AnoGAN objective."""
    if config["training"].get("gan_type", "bce") != "bce":
        raise ValueError("this repository supports the released BCE model only")
    generator.train()
    discriminator.train()
    optimizer_g, optimizer_d = optimizers
    batch_size = int(config["training"]["batch_size"])
    latent_dim = int(config["model"]["latent_dim"])
    label_smoothing = float(
        config["training"].get("label_smoothing", 0.0))
    gradient_clip = float(
        config["training"].get("gradient_clip", 0.0))
    random_generator = torch.Generator(device=device).manual_seed(
        int(seed) * 1000003 + int(epoch))
    order = np.random.RandomState(seed + epoch).permutation(len(features))
    total_batches = int(math.ceil(len(order) / float(batch_size)))
    criterion = nn.BCEWithLogitsLoss()
    sums = {
        "generator_loss": 0.0,
        "discriminator_loss": 0.0,
        "generator_grad_norm": 0.0,
        "discriminator_grad_norm": 0.0,
    }
    generated_moments = []
    seen = 0
    for batch_index, start in enumerate(range(0, len(order), batch_size), 1):
        indices = order[start:start + batch_size]
        real = torch.from_numpy(features[indices]).to(device)
        batch = len(real)

        optimizer_d.zero_grad(set_to_none=True)
        latent = torch.randn(
            batch, latent_dim, device=device, generator=random_generator)
        detached_fake = generator(latent).detach()
        real_target = torch.full(
            (batch,), 1.0 - label_smoothing, device=device)
        fake_target = torch.zeros(batch, device=device)
        discriminator_loss = (
            criterion(discriminator(real), real_target)
            + criterion(discriminator(detached_fake), fake_target))
        discriminator_loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), gradient_clip)
        discriminator_grad_norm = _gradient_norm(discriminator)
        optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        latent = torch.randn(
            batch, latent_dim, device=device, generator=random_generator)
        fake = generator(latent)
        generator_loss = criterion(
            discriminator(fake), torch.ones(batch, device=device))
        generator_loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), gradient_clip)
        generator_grad_norm = _gradient_norm(generator)
        optimizer_g.step()

        seen += batch
        sums["generator_loss"] += float(generator_loss.detach()) * batch
        sums["discriminator_loss"] += (
            float(discriminator_loss.detach()) * batch)
        sums["generator_grad_norm"] += generator_grad_norm * batch
        sums["discriminator_grad_norm"] += (
            discriminator_grad_norm * batch)
        with torch.no_grad():
            generated_moments.append((
                float(fake.mean()), float(fake.var(unbiased=False))))
        if status_callback:
            status_callback(batch_index, total_batches, seen)

    result = {
        name: value / max(seen, 1) for name, value in sums.items()}
    result["generator_output_mean"] = float(np.mean(
        [value[0] for value in generated_moments]))
    result["generator_output_variance"] = float(np.mean(
        [value[1] for value in generated_moments]))
    result["real_feature_variance"] = float(np.var(features))
    return result
