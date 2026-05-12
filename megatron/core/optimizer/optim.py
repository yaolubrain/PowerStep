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
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr, gamma, beta, wd = group['lr'], group['gamma'], group['beta'], group['weight_decay']
            G = group['block_size']

            for p in group['params']:
                if p.grad is None: continue
                                
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
                # We unsqueeze scales to [num_blocks, 1] for broadcasting
                m_prev = m_blocks_int8.to(p.dtype) * (scales.unsqueeze(1) / 127.0)

                # Accumulate: m_t = gamma * m_t-1 + g_t
                m_curr = m_prev.mul(gamma).add(g_blocks)

                # Re-quantize and Update Scales
                # Use max(dim=1) to get the absolute maximum of each block
                new_scales = m_curr.abs().max(dim=1, keepdim=True)[0]
                scales.copy_(new_scales.squeeze()) # Update the stored scales
                
                q_factor = 127.0 / (new_scales + 1e-12)
                m_updated_int8 = m_curr.mul(q_factor).round().clamp(-128, 127).to(torch.int8)
                
                # Save updated INT8 momentum (removing padding)
                m_int8.view(-1).copy_(m_updated_int8.view(-1)[:numel])

                # Apply Signed Power Transform
                m_curr = m_curr.sign() * torch.pow(m_curr.abs(), beta)
                
                # Copy the transformed updates back to the gradient buffer
                grad.copy_(m_curr.view(-1)[:numel])
                
                # Weight Update
                if wd != 0:
                    grad.add_(p, alpha=wd)
                
                p.add_(grad, alpha=-lr)
            
        return loss


class PBSGD(Optimizer):   
    def __init__(self, params, lr, gamma=0.9, beta=0.1, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")        
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (momentum): {gamma}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta, weight_decay=0.0)
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
                    state['exp_avg_sq'] = torch.zeros_like(p)

                grad = grad.sign() * (grad.abs()).pow(beta)
                exp_avg = state['exp_avg']
                exp_avg.mul_(gamma).add_(grad)
                update = exp_avg
                
                if wd != 0:
                    update.add_(p, alpha=wd)

                p.add_(update, alpha=-lr)

        return loss        
    


class AdamS(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, 
                 weight_decay=0.1):
       
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, 
                        eps=eps, weight_decay=weight_decay, step_t=1)
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
    