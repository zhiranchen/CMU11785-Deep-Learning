from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

'''
If you plan to use the code skeleton from therecitation pass model.parameters() 
from train loop.model here is an object of Seq2Seq class.
'''

def plot_grad_flow(named_parameters):    
    '''Plots the gradients flowing through different layers in the net during training.    
       Can be used for checking for possible gradient vanishing / exploding problems.        
       Usage: Plug this function in Trainer class after loss.backwards() as     
       "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''    
    ave_grads = []    
    max_grads= []    
    layers = []    
    for n, p in named_parameters:        
        if(p.requires_grad) and ("bias" not in n):            
            if(p is not None):                
                layers.append(n)                
                ave_grads.append(p.grad.abs().mean())                
                max_grads.append(p.grad.abs().max())
    plt.clf()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")    
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")    
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )    
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")    
    plt.xlim(left=0, right=len(ave_grads))    
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions    
    plt.xlabel("Layers")
    plt.ylabel("average gradient")    
    plt.title("Gradient flow")    
    #plt.tight_layout()    
    plt.grid(True)    
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])    
    return plt