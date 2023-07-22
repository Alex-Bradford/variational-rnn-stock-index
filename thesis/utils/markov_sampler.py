from __future__ import absolute_import

import torch
import torch.nn as nn


class MarkovSamplingLoss(object):

    def __init__(self, model, samples) -> None:
        self.model = model
        self.samples = samples
        self.mse = nn.MSELoss()
        self.llhood = nn.GaussianNLLLoss(reduction='sum')

    def __call__(self, X, y, num_batches, testing=False):
        
        # Validate input shape
        assert len(X.shape)==3, f"Expected input to be 3-dim, got {len(X.shape)}"
        batch_size, seq_size, feat_size = X.shape

        # Define output tensors
        outputs = torch.zeros(self.samples, batch_size, self.model.output_dim)
        log_prior = torch.tensor(0, dtype=torch.float)
        log_variational_posterior = torch.tensor(0, dtype=torch.float)

        # Sample and compute pdfs
        for s in range(self.samples):

            outputs[s] = self.model(X, sampling=True,testing=testing)
            
            if testing:
                continue
            
            log_prior += self.model.log_prior()
            log_variational_posterior += self.model.log_variational_posterior()
        
        # Return output if testing
        if testing:
            return outputs

        # print('testing:',testing,'model.training:',self.model.training,'log_prior:',self.model.log_prior())
        # Log prior, variational posterior and likelihood
        var = torch.ones(batch_size,self.model.output_dim)
        negative_log_likelihood = self.mse(outputs.mean(0), y)
        negative_log_likelihood2 = self.llhood(outputs.mean(0), y, var)*self.samples

        # fixes:
        #   Use log likelihood instead of MSE
        #   Use reduction='sum' in the log likelihood to match the log_var_posterior and log_prior
        #   multiply the log likelihood by the number of samples, also to match the log_var_posterior and log_prior

        loss = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood
        loss2 = (log_variational_posterior - log_prior*(num_batches*negative_log_likelihood)) / num_batches
        loss3 = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood2

        term1 = (log_variational_posterior)/num_batches
        term2 = -log_prior/num_batches
        term3 = negative_log_likelihood2

        # print("loss:",loss)
        # print("loss2:", loss2)
        
        return loss3, outputs, term1,term2,term3
