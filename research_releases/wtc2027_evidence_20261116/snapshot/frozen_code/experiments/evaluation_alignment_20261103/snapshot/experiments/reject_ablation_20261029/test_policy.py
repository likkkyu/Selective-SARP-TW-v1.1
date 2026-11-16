import unittest
import torch
from run import apply_rule

class PolicyTest(unittest.TestCase):
    def test_rules_and_gradients(self):
        mask = torch.tensor([[[False, False, True, False]],
                             [[False, True, True, False]],
                             [[False, True, True, True]],
                             [[True, False, False, True]]])
        logits = torch.zeros(4, 1, 4, requires_grad=True)
        masked = logits.masked_fill(mask, float('-inf'))
        out = apply_rule(masked, mask)
        self.assertTrue(torch.isneginf(out[0, 0, -1]))
        self.assertEqual(out[1].argmax().item(), 3)
        self.assertEqual(out[2].argmax().item(), 0)
        self.assertTrue(torch.isneginf(out[mask]).all())
        prob = out.softmax(-1)
        self.assertTrue(torch.isfinite(prob).all())
        prob[..., 0].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

if __name__ == '__main__':
    unittest.main()
