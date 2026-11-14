"""Validate patience semantics, learning-rate floor, and scheduler recovery."""
import copy
import unittest
import torch
from colab.scheduler_runner import ScheduleControl

SETTINGS=dict(lr_factor=.5,lr_patience=2,min_delta=1e-5,min_lr=1e-5,early_stop_patience=6)

class SchedulerTests(unittest.TestCase):
    def make(self, **kwargs):
        opt=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.001)
        return opt,ScheduleControl(opt,{**SETTINGS,**kwargs})

    def test_patience_early_stop_and_first_reduction(self):
        opt,s=self.make();rates=[];stops=[]
        for _ in range(7):
            rates.append(opt.param_groups[0]['lr']);stops.append(s.step(.7))
        self.assertEqual(rates,[.001,.001,.001,.001,.0005,.0005,.0005])
        self.assertEqual(stops,[False]*6+[True])
        self.assertEqual(opt.param_groups[0]['lr'],.00025)

    def test_scheduler_and_early_stop_resume(self):
        opt,s=self.make()
        for loss in [.7,.7,.7,.7]: s.step(loss)
        new_opt,new_s=self.make()
        new_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        new_s.load_state_dict(copy.deepcopy(s.state_dict()))
        for loss in [.699999,.69,.69,.69,.69]:
            self.assertEqual(s.step(loss),new_s.step(loss))
            self.assertEqual(opt.param_groups[0]['lr'],new_opt.param_groups[0]['lr'])
            self.assertEqual(s.stale,new_s.stale)

    def test_absolute_tolerance_and_lr_floor(self):
        opt,s=self.make(lr_patience=0,early_stop_patience=100)
        s.step(.7);s.step(.7-1e-6)
        self.assertEqual(s.stale,1)
        for _ in range(20): s.step(.7)
        self.assertEqual(opt.param_groups[0]['lr'],1e-5)
        s.step(.69);self.assertEqual(s.stale,0)
        with self.assertRaises(ValueError): s.step(float('nan'))

if __name__=='__main__':unittest.main()
