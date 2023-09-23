import torch
class synth_expert:
    def __init__(self, k1=None, k2=None, n_classes=None, S=None, p_in=None, p_out=None,device='cuda:0'):
        ''' 
        class to model the non-overlapping synthetic experts
        
        The expert predicts correctly for classes k1 (inclusive) to k2 (exclusive), and 
        random across the total number of classes for other classes outside of [k1, k2).

        For example, an expert could be correct for classes 2 (k1) to 4 (k2) for CIFAR-10.

        '''
        self.k1 = k1
        self.k2 = k2
        self.p_in = p_in if p_in is not None else 1.0   #IGNORE
        self.p_out = p_out if p_out is not None else 1/n_classes #IGNORE
        self.n_classes = n_classes
        self.S = S # list : set of classes where the oracle predicts #IGNORE
        self.device=device
	
    def predict(self, input, labels): 	# expert correct in [k1,k2) classes else random across all the classes	
        out=torch.zeros(labels.shape).long().to(self.device)
        batch_size = labels.size()[0]  # batch_size
        # print("MASK-",labels.shape,self.k1)
        mask=torch.logical_and(labels<self.k2 , labels>=self.k1)
        out[mask]=labels[mask]
        rand_value=torch.randint(0, self.n_classes, labels.shape).to(self.device)
        out[~mask]=rand_value[~mask]


        return out


