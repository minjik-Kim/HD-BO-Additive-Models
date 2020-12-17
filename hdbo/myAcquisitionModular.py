import numpy as np
import GPyOpt.util.stats as gpstats
from GPyOpt.acquisitions.base import AcquisitionBase

class MyAcquisitionModular(AcquisitionBase):
    
    def __init__(self, model, optimizer, domain):
        super(MyAcquisitionModular, self).__init__(model, None, optimizer)

    def optimize(self, duplicate_manager):
        # Only the first arg will be used sequential
        return self.optimizer.optimize(f=self.model.cfn, f_df=self.model.cfn.acq_f_df)[0], None