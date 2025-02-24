import torch

class EarlyStopper:
    '''
    function for Early Stopping of learning epochs 
    
    if loss value is not step until patience value 
    your machine will stop after patience
    
    class args:
            patience: patience value of step -> int
            min_delta: min change amount for early stop -> int
    '''
    def __init__(self, patience:int=3, min_delta:int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.save_counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, model, validation_loss:float, name='test.pth', mode=True):
        if validation_loss < self.min_validation_loss:
            if self.counter >= 1:
                self.save_counter += 1
                print('early stoper save a model! count : {}'.format(self.save_counter))
                torch.save(model.state_dict(), name)  
            self.counter = 0
            self.min_validation_loss = validation_loss
            
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1  
            if self.counter >= self.patience and mode:
                print('early stoper run!')
                return True
            
        return False