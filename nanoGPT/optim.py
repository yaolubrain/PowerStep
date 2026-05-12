import torch
import torch.nn.functional as F
from torch.optim import Optimizer


class PowerStep(Optimizer):
    def __init__(self, params, lr, gamma=0.9, beta=0.1, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")        
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (momentum): {gamma}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)                    

                exp_avg = state['exp_avg']                
                exp_avg.mul_(gamma).add_(grad)
                                                              
                update = exp_avg.sign() * exp_avg.abs().pow(beta)
                
                if wd != 0:
                    update.add_(p, alpha=wd)

                p.add_(update, alpha=-lr)
                
        return loss


class PowerStepInt8(Optimizer):
    def __init__(self, params, lr, gamma=0.9, beta=0.1, weight_decay=0.0, block_size=128):
        defaults = dict(lr=lr, gamma=gamma, beta=beta, 
                        weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): 
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            wd = group['weight_decay']
            G = group['block_size']

            for p in group['params']:
                if p.grad is None: 
                    continue
                                
                state = self.state[p]
                
                # Initialize state if empty
                if len(state) == 0:
                    state['m_int8'] = torch.zeros_like(p, dtype=torch.int8)                  
                    num_blocks = (p.numel() + G - 1) // G
                    state['scales'] = torch.zeros(num_blocks, device=p.device, dtype=p.dtype)

                # Retrieve state variables
                m_int8 = state['m_int8']
                scales = state['scales']
                grad = p.grad.view(-1)
                numel = grad.numel()
                p_flat = p.view(-1)  # Flatten parameter for consistent operations
            
                # Vectorized Padding & Reshaping
                pad_size = (G - (numel % G)) % G
                if pad_size > 0:
                    g_padded = F.pad(grad, (0, pad_size))
                    m_int8_padded = F.pad(m_int8.view(-1), (0, pad_size))
                else:
                    g_padded = grad
                    m_int8_padded = m_int8.view(-1)

                g_blocks = g_padded.view(-1, G)
                m_blocks_int8 = m_int8_padded.view(-1, G)

                # Dequantize: m_prev = (m_int8 * scales) / 127
                m_prev = m_blocks_int8.to(p.dtype) * (scales.unsqueeze(1) / 127.0)

                # Accumulate: m_t = gamma * m_t-1 + g_t
                m_curr = m_prev.mul(gamma).add(g_blocks)

                # Re-quantize and Update Scales
                new_scales = m_curr.abs().max(dim=1, keepdim=True)[0]
                scales.copy_(new_scales.squeeze())
                
                q_factor = 127.0 / (new_scales + 1e-12)
                m_updated_int8 = m_curr.mul(q_factor).round().clamp(-128, 127).to(torch.int8)
                
                # Save updated INT8 momentum (removing padding)
                m_int8.view(-1).copy_(m_updated_int8.view(-1)[:numel])

                # Apply Signed Power Transform
                update = m_curr.sign() * torch.pow(m_curr.abs(), beta)
                
                # Copy the transformed updates back to the gradient buffer
                grad.copy_(update.view(-1)[:numel])
                
                # Weight Update (using flattened views)
                if wd != 0:
                    # Decoupled weight decay (AdamW-style)
                    p_flat.mul_(1 - lr * wd)
                    p_flat.add_(grad, alpha=-lr)
                else:
                    p_flat.add_(grad, alpha=-lr)
            
        return loss


class SignSGD(Optimizer):
    def __init__(self, params, lr, gamma=0.9, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")        
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (momentum): {gamma}")

        defaults = dict(lr=lr, gamma=gamma, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)                    

                exp_avg = state['exp_avg']                
                exp_avg.mul_(gamma).add_(grad)
                                                              
                update = exp_avg.sign()
                
                if wd != 0:
                    update.add_(p, alpha=wd)

                p.add_(update, alpha=-lr)
                
        return loss


class AdamWInt8(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, block_size=128):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']            
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            G = group['block_size']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamWInt8 does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize INT8 state for momentum m and v
                if len(state) == 0:
                    state['step'] = 0
                    # Store both m and v in INT8 format
                    state['m_int8'] = torch.zeros_like(p, dtype=torch.int8)
                    state['v_int8'] = torch.zeros_like(p, dtype=torch.int8)
                    
                    numel = p.numel()
                    num_blocks = (numel + G - 1) // G
                    state['m_scales'] = torch.zeros(num_blocks, device=p.device, dtype=torch.float32)
                    state['v_scales'] = torch.zeros(num_blocks, device=p.device, dtype=torch.float32)
                
                state['step'] += 1
                step = state['step']
                
                m_int8 = state['m_int8']
                v_int8 = state['v_int8']
                m_scales = state['m_scales']
                v_scales = state['v_scales']
                
                numel = p.numel()
                
                # Handle padding for block-wise operations
                pad_size = (G - (numel % G)) % G
                if pad_size > 0:
                    grad_flat = grad.view(-1)
                    grad_padded = F.pad(grad_flat, (0, pad_size))
                    m_int8_padded = F.pad(m_int8.view(-1), (0, pad_size))
                    v_int8_padded = F.pad(v_int8.view(-1), (0, pad_size))
                else:
                    grad_padded = grad.view(-1)
                    m_int8_padded = m_int8.view(-1)
                    v_int8_padded = v_int8.view(-1)
                
                # Reshape into blocks
                g_blocks = grad_padded.view(-1, G)
                m_blocks_int8 = m_int8_padded.view(-1, G)
                v_blocks_int8 = v_int8_padded.view(-1, G)
                
                # 1. Dequantize m and v
                m_prev = m_blocks_int8.to(torch.float32) * (m_scales.unsqueeze(1) / 127.0)
                v_prev = v_blocks_int8.to(torch.float32) * (v_scales.unsqueeze(1) / 127.0)
                
                # 2. Update biased first moment estimate
                m_curr = m_prev.mul(beta1).add(g_blocks, alpha=1 - beta1)
                
                # 3. Update biased second raw moment estimate
                v_curr = v_prev.mul(beta2).addcmul(g_blocks, g_blocks, value=1 - beta2)
                
                # 4. Re-quantize m and v and update scales
                # For m
                new_m_scales = m_curr.abs().max(dim=1, keepdim=True)[0]
                m_scales.copy_(new_m_scales.squeeze())
                q_factor_m = 127.0 / (new_m_scales + 1e-12)
                m_updated_int8 = m_curr.mul(q_factor_m).round().clamp(-128, 127).to(torch.int8)
                
                # For v (always non-negative)
                new_v_scales = v_curr.max(dim=1, keepdim=True)[0]
                v_scales.copy_(new_v_scales.squeeze())
                q_factor_v = 127.0 / (new_v_scales + 1e-12)
                v_updated_int8 = v_curr.mul(q_factor_v).round().clamp(-128, 127).to(torch.int8)
                
                # Copy back (without padding)
                m_int8.view(-1).copy_(m_updated_int8.view(-1)[:numel])
                v_int8.view(-1).copy_(v_updated_int8.view(-1)[:numel])
                
                # 5. Compute bias-corrected estimates
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                m_hat = m_curr / bias_correction1
                v_hat = v_curr / bias_correction2
                
                # 6. Compute update
                denom = v_hat.sqrt().add_(eps)
                update = m_hat / denom
                
                # 7. Apply weight decay (AdamW style - decoupled)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # 8. Apply update to parameters
                p.view(-1).add_(update.view(-1)[:numel], alpha=-lr)
                            
        return loss

class AdamS(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, 
                 weight_decay=0.1):
       
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, 
                        eps=eps, weight_decay=weight_decay, step_t=0)
        super().__init__(params, defaults)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']            
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']            
            
            group['step_t'] += 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad                            
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                
                m = state['m']                
                m.mul_(beta1).add_(grad, alpha=1-beta1)            
                v = beta2*m.pow(2) + (1-beta2)*grad.pow(2)
      
                m_hat = m / (1 - beta1 ** group['step_t'])
                v_hat = v / (1 - beta2 ** group['step_t'])
                
                sqrt_v = torch.sqrt(v_hat) + eps
                        
                p.mul_(1 - lr * wd)
                p.add_(m_hat / sqrt_v, alpha=-lr)
                            
        return loss    

class PBSGD(Optimizer):   
    def __init__(self, params, lr, gamma=0.9, beta=0.1, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")        
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (momentum): {gamma}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                grad = grad.sign() * (grad.abs()).pow(beta)
                exp_avg = state['exp_avg']
                exp_avg.mul_(gamma).add_(grad)
                update = exp_avg
                
                if wd != 0:
                    update.add_(p, alpha=wd)

                p.add_(update, alpha=-lr)

        return loss
    

class Stacey(Optimizer):
    """Stacey(p,2) optimizer as described in the paper.
    
    Implements Algorithm 1 from:
    "Stacey: Promoting Stochastic Steepest Descent via Accelerated ℓp-Smooth 
     Nonconvex Optimization" (Luo et al., 2025)
    
    Args:
        lr: learning rate (η_t in paper)
        power: p value for ℓp norm geometry (default 11 gives β=0.1 in PowerStep terms)
        beta1: EMA coefficient for dual correction term c (default: 0.9)
        beta2: EMA coefficient for momentum buffer m (default: 0.95)
        eps: stabilization constant for signed power transform (default: 1e-8)
        alpha: primal-dual interpolation factor (default: 0.5)
        tau: weight for dual vs primal direction (default: 0.0)
        weight_decay: decoupled weight decay (λ in paper) (default: 0.0)
    """
    
    def __init__(self, params, lr=0.01, power=11, beta1=0.9, beta2=0.95, 
                 eps=1e-8, alpha=0.5, tau=0.0, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 (EMA coefficient for dual): {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 (EMA coefficient for momentum): {beta2}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha (primal-dual interpolation): {alpha}")

        defaults = dict(lr=lr, power=power, beta1=beta1, beta2=beta2, eps=eps, 
                        alpha=alpha, tau=tau, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            power = group['power']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            alpha = group['alpha']
            tau = group['tau']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)   # EMA momentum buffer
                    state['z'] = torch.zeros_like(p)   # Primal-dual accumulator

                m = state['m']
                z = state['z']
                state['step'] += 1

                # Step 1: Update EMA momentum: m_{t+1} = β₂m_t + (1-β₂)g̃_t
                # CORRECTED: Added (1 - beta2) factor to match paper exactly
                m.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Step 2: Compute correction term: c_{t+1} = β₁m_t + (1-β₁)g̃_t
                # Note: Uses current m after update and current grad
                c = m.mul(beta1).add(grad, alpha=1 - beta1)

                # Step 3: Stabilized signed power transform on c
                # y_{t+1} = c / (|c|^{(p-2)/(p-1)} + ε)
                # This is the signed power transform: sign(c) * |c|^{1/(p-1)}
                # but written in stabilized form to avoid numerical issues
                y = c / (c.abs().pow((power - 2) / (power - 1)) + eps)

                # Step 4: Update primal-dual accumulator: z_{t+1} = z_t - α·c_{t+1}
                # Paper says: z_{t+1} = z_t - α·c_{t+1}
                # Note: some implementations flip sign convention but this is most direct
                z.mul_(1.0).add_(c, alpha=-alpha)

                # Step 5: Compute final update direction
                # u_t = -η_t[τ·z_{t+1} + (1-τ)·y_{t+1}]
                update = z.mul(tau).add(y, alpha=1 - tau)

                # Step 6: Apply weight decay (decoupled): θ -= η_t·λ·θ_t
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Step 7: Apply gradient-based update: θ += u_t (u_t already includes -η_t)
                p.add_(update, alpha=-lr)

        return loss