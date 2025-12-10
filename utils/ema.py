#!/usr/bin/env python3
"""
Exponential Moving Average (EMA) for Model Weights

Maintains a shadow copy of model parameters using exponential moving average.
EMA weights typically provide better generalization and smoother predictions.

Usage:
    ema = ModelEMA(model, decay=0.999, update_after_step=100)
    
    # Training loop
    for step, (X, Y) in enumerate(train_loader):
        loss = train_step(model, X, Y)
        optimizer.step()
        ema.update(model, step)
    
    # Validation with EMA weights
    with ema.average_parameters():
        val_loss = validate(model, val_loader)

Author: Energy-Efficient STGNN Project
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that are updated using:
        ema_param = decay * ema_param + (1 - decay) * model_param
    
    Args:
        model: PyTorch model to track
        decay: EMA decay rate (default: 0.999). Higher = smoother averaging
        update_after_step: Start EMA updates after this many steps (default: 100)
        device: Device to store EMA parameters (default: same as model)
    """
    
    def __init__(self, model, decay=0.999, update_after_step=100, device=None):
        self.decay = decay
        self.update_after_step = update_after_step
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow copy of model parameters
        self.shadow_params = {}
        self._register_parameters(model)
        
        self.num_updates = 0
        
    def _register_parameters(self, model):
        """Store initial copy of model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach().to(self.device)
    
    def update(self, model, step=None):
        """
        Update EMA parameters with current model parameters.
        
        Args:
            model: Current model
            step: Current training step (optional, uses internal counter if None)
        """
        if step is not None:
            current_step = step
        else:
            current_step = self.num_updates
        
        # Only update after warmup period
        if current_step < self.update_after_step:
            return
        
        # Dynamic decay adjustment (optional - starts lower, increases to target)
        # decay = min(self.decay, (1 + current_step) / (10 + current_step))
        decay = self.decay
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # EMA update: shadow = decay * shadow + (1 - decay) * current
                    self.shadow_params[name].mul_(decay).add_(
                        param.data.to(self.device), alpha=1 - decay
                    )
        
        self.num_updates += 1
    
    def apply_shadow(self, model):
        """
        Apply EMA weights to model (replaces current weights).
        
        Args:
            model: Model to update
            
        Returns:
            dict: Backup of original parameters (for restoration)
        """
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    backup[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name])
        return backup
    
    def restore(self, model, backup):
        """
        Restore original model weights from backup.
        
        Args:
            model: Model to restore
            backup: Backup dictionary from apply_shadow()
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    
    @contextmanager
    def average_parameters(self, model=None):
        """
        Context manager to temporarily use EMA weights.
        
        Usage:
            with ema.average_parameters(model):
                # model now uses EMA weights
                validate(model, val_loader)
            # model restored to original weights
        
        Args:
            model: Model to temporarily modify (if None, no-op for backward compat)
        """
        if model is None:
            yield
            return
            
        backup = self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model, backup)
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow_params': self.shadow_params,
            'num_updates': self.num_updates,
            'decay': self.decay,
            'update_after_step': self.update_after_step,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow_params = state_dict['shadow_params']
        self.num_updates = state_dict.get('num_updates', 0)
        self.decay = state_dict.get('decay', self.decay)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
    
    def to(self, device):
        """Move EMA parameters to device."""
        self.device = device
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)
        return self


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_ema(model, config=None):
    """
    Create ModelEMA instance from config.
    
    Args:
        model: PyTorch model
        config: Config module (uses utils.config if None)
        
    Returns:
        ModelEMA instance or None if EMA disabled
    """
    if config is None:
        from utils import config
    else:
        config = config
    
    if not getattr(config, 'USE_EMA', False):
        return None
    
    return ModelEMA(
        model,
        decay=getattr(config, 'EMA_DECAY', 0.999),
        update_after_step=getattr(config, 'EMA_UPDATE_AFTER_STEP', 100)
    )


if __name__ == "__main__":
    # Test EMA functionality
    print("Testing ModelEMA...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Initialize EMA
    ema = ModelEMA(model, decay=0.999, update_after_step=0)
    
    # Simulate training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nInitial model weight (first layer):")
    print(model[0].weight[0, :5])
    
    # Training steps
    for step in range(10):
        # Dummy forward/backward
        loss = model(torch.randn(32, 10)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update EMA
        ema.update(model, step)
    
    print("\nAfter 10 steps - model weight:")
    print(model[0].weight[0, :5])
    
    print("\nAfter 10 steps - EMA weight:")
    print(ema.shadow_params['0.weight'][0, :5])
    
    # Test context manager
    print("\nTesting context manager...")
    with ema.average_parameters(model):
        print("Inside context (EMA weights):")
        print(model[0].weight[0, :5])
    
    print("\nOutside context (restored):")
    print(model[0].weight[0, :5])
    
    print("\n✅ EMA test completed successfully!")
