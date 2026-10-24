"""Meaningful numerical checks; skipped if the user has not installed ML dependencies."""

import importlib.util
import os
import unittest

AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('numpy','torch','sklearn'))
if AVAILABLE:
    import numpy as np
    import torch
    from fleetpm.features import Preprocessor,grouped_folds,observation_gaps,rolling_statistics,split_engines,window_packet,STAT_DIM
    from fleetpm.models import HybridRUL
    from fleetpm.learning import tensor_batch,model_forward


@unittest.skipUnless(AVAILABLE,'Optional PyTorch research stack is not installed; neural/CUDA checks not run')
class NumericalTests(unittest.TestCase):
    def engines(self):
        return {unit:{'unit':unit,'cycles':np.arange(1,41),'raw':np.column_stack([np.tile([0,0,100],(40,1)),np.arange(40)[:,None]*np.ones((1,21))*.1+unit]).astype(np.float32)} for unit in range(1,21)}

    def test_engine_partitions_and_fold_isolation(self):
        split=split_engines(self.engines());sets=[set(v) for v in split.values()]
        self.assertEqual(set.union(*sets),set(range(1,21)))
        for i,left in enumerate(sets):
            for right in sets[i+1:]:self.assertFalse(left&right)
        seen=[]
        for _,fit,held in grouped_folds(split['fit'],3,42):self.assertFalse(set(fit)&set(held));seen.extend(held)
        self.assertEqual(sorted(seen),split['fit'])

    def test_preprocessing_and_features_preserve_causality(self):
        engines=self.engines();processor=Preprocessor('FD001').fit(engines,list(range(1,11)))
        before=window_packet(processor.transform(engines[20]),29)
        engines[20]['raw'][30:,3:]+=10000
        after=window_packet(processor.transform(engines[20]),29)
        np.testing.assert_array_equal(before['x'],after['x']);np.testing.assert_array_equal(before['stats'],after['stats'])
        self.assertEqual(processor.fit_ids,list(range(1,11)))

    def test_statistics_and_gap_continuation(self):
        result=rolling_statistics(np.ones((1,21),np.float32),np.ones((1,21),bool));self.assertEqual(result.shape,(STAT_DIM,));self.assertTrue(np.isfinite(result).all())
        gaps=observation_gaps(np.zeros((2,21),bool),np.full(21,10));self.assertAlmostEqual(float(gaps[0,0]),11/30,places=6)

    def test_network_padding_trusted_merge_and_finite_backward(self):
        torch.manual_seed(42);model=HybridRUL().eval()
        x=torch.randn(2,30,66);x[:,:,24:45]=1
        stats=torch.zeros(2,STAT_DIM);q=torch.ones(2,3);lengths=torch.tensor([15,30]);valid=torch.arange(30)[None,:]<lengths[:,None]
        packet,_=model.repair_packet(x);torch.testing.assert_close(packet[:,:,:21],x[:,:,:21])
        pred,_=model(x,stats,q,lengths,valid);self.assertTrue(torch.isfinite(pred).all());self.assertEqual(pred.shape,(2,))
        changed=x.clone();changed[0,15:,:]=100
        later,_=model(changed,stats,q,lengths,valid);torch.testing.assert_close(pred[0],later[0])
        loss=pred.sum();loss.backward();self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))

    @unittest.skipUnless(os.environ.get('FLEET_TEST_CUDA')=='1','CUDA numerical check is explicit opt-in')
    def test_cuda_amp_forward_backward(self):
        self.assertTrue(torch.cuda.is_available(),'Explicit CUDA validation requested but runtime unavailable')
        model=HybridRUL().cuda();x=torch.zeros(2,30,66,device='cuda');x[:,:,24:45]=1
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast('cuda',dtype=dtype):
            point,_=model(x,torch.zeros(2,STAT_DIM,device='cuda'),torch.ones(2,3,device='cuda'),torch.tensor([30,30],device='cuda'),torch.ones(2,30,dtype=torch.bool,device='cuda'))
            loss=point.mean()
        self.assertTrue(torch.isfinite(loss));loss.backward()
