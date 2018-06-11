from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd as autograd

from torch import Tensor
from torch.nn import Module

# This module implements KF block approximation for the hessian
# http://proceedings.mlr.press/v70/botev17a/botev17a.pdf

# KFC is used for convolutional layers.
# (https://arxiv.org/pdf/1602.01407.pdf).


class ArchitectureNotSupported(Exception):
    """Exception raised when KF block approximation of the Hessian cannot
    be computed. Only architectures that alternate linear layers and
    element-wise transfer functions are supported.

    """
    pass


# -- Functions that inspects layer type

def is_convolutional(module: Module) -> bool:
    return isinstance(module, nn.Conv2d)


def is_linear(module: Module) -> bool:
    return isinstance(module, nn.Linear)


def is_activation(module: Module) -> bool:
    # TODO: improve this (it sucks now!)
    not_elmentwise = [nn.Softmax, nn.Softmax2d, nn.Softmin]
    if any(isinstance(module, Activation) for Activation in not_elmentwise):
        return False
    return type(module).__module__ == 'torch.nn.modules.activation'


class KFHessianProduct(object):

    def __init__(self,
                 inputs_cov: Dict[str, List[Tensor]],
                 outputs_hess: Dict[str, List[Tensor]]):

        if len(inputs_cov) != len(outputs_hess):
            raise ValueError("Lists should be equal in length.")

        self.factors = dict({})  # type: Dict[str, List[Tuple[Tensor, Tensor]]]
        for module_name, i_cov_lst in inputs_cov.items():
            o_hess_lst = outputs_hess[module_name]
            factors_lst = self.factors.setdefault(module_name, [])
            for i_cov, o_hess in zip(i_cov_lst, o_hess_lst):
                factors_lst.append((i_cov.clone().detach(), o_hess.clone().detach()))

    def __compute_product(self, module_name: str, weight: Tensor, bias: Tensor):
        out_no = weight.size(0)
        weight, bias = weight.view(out_no, -1), bias.view(out_no, 1)
        params = torch.cat([weight, bias], dim=1)
        loss, prods_no = params.new_zeros(1), 0
        for i_cov, o_hess in self.factors[module_name]:
            loss += (o_hess @ params @ i_cov).sum()
            prods_no += 1
        loss /= prods_no
        return loss

    def hessian_product_loss(self, vector: Dict[str, Tensor]) -> None:
        loss = None
        for module_name in self.factors:
            weight = vector[f"{module_name:s}.weight"]
            bias = vector[f"{module_name:s}.bias"]
            layer_loss = self.__compute_product(module_name, weight, bias)
            loss = layer_loss if loss is None else (layer_loss + loss)
        return loss


class KroneckerFactored(nn.Module):

    ACTIVATION = 1
    LINEAR = 2
    CONVOLUTIONAL = 3
    OTHER = 4

    FORWARD = 1
    BACKWARD = 2
    DONE = 3

    def __init__(self,
                 do_checks: bool=False,
                 use_fisher: bool=True,
                 verbose: bool=False,
                 average_factors: bool=True) -> None:
        super(KroneckerFactored, self).__init__()
        self.__my_handles = []
        self.__kf_mode = False  # One must activate this
        self.__use_fisher = use_fisher
        self.__do_checks = do_checks
        self.__verbose = verbose
        self.__average_factors = average_factors

        # Per KF computation (persistent over several fwd+bwd's)
        self.__outputs_hess = dict({})  # type: Dict[str, List[Tensor]]
        self.__inputs_cov = dict({})  # type: Dict[str, List[Tensor]]
        self.__module_names = dict({})  # type: Dict[int, str]
        self.__batches_no = 0

        # Per fwd+bwd pass
        self.__df_dx = dict({})  # type: Dict[str, Tensor]
        self.__d2f_dx2 = dict({})  # type: Dict[str, Tensor]
        self.__prev_layer = None  # type: int
        self.__prev_layer_name = None  # type: str
        self.__phase = None  # type: int
        self.__layer_idx = None  # type: int
        self.__last_linear = None  # type: int
        self.__conv_special_inputs = dict({})
        self.__next_parametric = None
        self.__maybe_exact = None
        self.__last_hessian = None
        self.__conv_special_inputs = dict({})
        self.__next_outputs_hessian = None

    def __reset_state(self):
        """This should be called whenever a new hessian is needed"""
        self.__soft_reset_state()
        self.__outputs_hess.clear()
        self.__inputs_cov.clear()
        self.__batches_no = 0
        self.__module_names.clear()
        for module_name, module in self.named_modules():
            self.__module_names[id(module)] = module_name

    def __soft_reset_state(self):
        """This should be called before each batch"""
        print("SOFT_RESET")
        self.__df_dx.clear()
        self.__d2f_dx2.clear()
        self.__prev_layer = None
        self.__prev_layer_name = "input"
        self.__phase = self.FORWARD
        self.__layer_idx = 0
        self.__last_linear = -1
        self.__conv_special_inputs.clear()
        self.__next_parametric = None  # type: Optional[Module]
        self.__maybe_exact = True
        self.__last_hessian = None
        self.__next_outputs_hessian = None

    @property
    def average_factors(self) -> bool:
        return self.__average_factors

    @average_factors.setter
    def average_factors(self, value: bool) -> None:
        self.__average_factors = value

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.__verbose = value

    @property
    def do_kf(self) -> bool:
        return self.__kf_mode

    @do_kf.setter
    def do_kf(self, value: bool) -> None:
        if (value and self.__kf_mode) or (not value and not self.__kf_mode):
            return
        self.__kf_mode = value
        if not value:
            self.__reset_state()
            self.__drop_hooks()
        else:
            self.__set_hooks()
            self.__reset_state()

    @property
    def output_hessian(self) -> Optional[Tensor]:
        return self.__last_hessian

    @output_hessian.setter
    def output_hessian(self, value: Tensor) -> None:
        if not torch.is_tensor(value) or \
           value.size() != self.__expected_output_hessian_size:
            raise ValueError
        if self.__phase != self.BACKWARD:
            raise Exception("Bad time to set the output_hessian")
        self.__last_hessian = value

    def __set_hooks(self):
        for module in self.modules():
            self.__my_handles.extend([
                module.register_forward_pre_hook(self._pre_hook),
                module.register_forward_hook(self._fwd_hook),
                module.register_backward_hook(self._bwd_hook)
            ])

    def __drop_hooks(self):
        for handle in self.__my_handles:
            handle.remove()
        self.__my_handles.clear()

    def _pre_hook(self, module, _inputs):
        """This hook only checks the architecture"""
        use_fisher = self.__use_fisher
        prev_layer = self.__prev_layer
        prev_name = self.__prev_layer_name
        crt_name = module._get_name()
        msg = f"Need Fisher for {prev_name:s} -> {crt_name:s}."

        if self.__verbose:
            print(f"[{self.__layer_idx:d}] {crt_name:s} before FWD")

        if isinstance(module, KroneckerFactored):
            self.__soft_reset_state()  # Starting a new fwd pass
        elif is_linear(module):
            if not (use_fisher or prev_layer is None or prev_layer == self.ACTIVATION):
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.LINEAR
        elif is_activation(module):
            if not (use_fisher or prev_layer == self.LINEAR):
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.ACTIVATION
        elif is_convolutional(module):
            if not use_fisher:
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.CONVOLUTIONAL
        else:
            if not use_fisher:
                raise ArchitectureNotSupported(msg)
            self.__prev_layer = self.OTHER
        self.__prev_layer_name = crt_name

    def _fwd_hook(self, module, inputs, output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} after FWD")

        if isinstance(module, KroneckerFactored):
            self.__kf_fwd_hook(module, inputs, output)
            return
        if is_linear(module):
            self.__linear_fwd_hook(module, inputs, output)
        elif is_activation(module):
            self.__activation_fwd_hook(module, inputs, output)
        elif is_convolutional(module):
            self.__conv_fwd_hook(module, inputs, output)
        elif not self.__use_fisher:
            raise ArchitectureNotSupported("You shouldn't be here!")

        self.__layer_idx += 1

    def _bwd_hook(self, module, grad_input, grad_output) -> None:
        if self.__verbose:
            crt_name = module._get_name()
            print(f"[{self.__layer_idx:d}] {crt_name:s} BWD")

        if isinstance(module, KroneckerFactored):
            return
        if is_linear(module):
            if self.__maybe_exact:
                self.__maybe_exact = self.__prev_layer is None or \
                    self.__prev_layer == self.ACTIVATION
            self.__linear_bwd_hook(module, grad_input, grad_output)
            self.__next_parametric = module
            self.__prev_layer = self.LINEAR
        elif is_activation(module):
            self.__maybe_exact = (self.__maybe_exact and self.__prev_layer == self.LINEAR)
            if self.__maybe_exact:
                self.__activation_bwd_hook(module, grad_input, grad_output)
            self.__prev_layer = self.ACTIVATION
        elif is_convolutional(module):
            self.__maybe_exact = False
            self.__conv_bwd_hook(module, grad_input, grad_output)
            self.__prev_layer = self.CONVOLUTIONAL
        else:
            self.__maybe_exact = False
            self.__prev_layer = self.OTHER

        if not self.__maybe_exact:
            self.__d2f_dx2.clear()
            self.__df_dx.clear()
            self.__next_outputs_hessian = None

        self.__layer_idx -= 1

        if self.__layer_idx < 0:
            self.__phase = self.DONE
            self.__batches_no += 1
            if self.__verbose:
                print(f"Done with this batch! We have {self.__batches_no:d}.")

    # Magic happens below

    def __kf_fwd_hook(self, _module, _inputs, _output):
        self.__phase = self.BACKWARD
        self.__layer_idx -= 1
        self.__maybe_exact = True
        self.__prev_layer = None

    # Hooks for linear layers

    def __linear_fwd_hook(self, module, inputs, output):
        assert self.__phase == self.FORWARD
        assert isinstance(inputs, tuple) and len(inputs) == 1
        assert isinstance(output, Tensor)

        x, = inputs  # extract from tuple
        b_sz = x.size(0)
        x1 = torch.cat([x, x.new_ones(b_sz, 1)], dim=1)  # add 1s
        inputs_cov = (x1.t() @ x1).detach_().div_(b_sz)
        layer_idx = self.__layer_idx
        module_name = self.__module_names[id(module)]

        lst = self.__inputs_cov.setdefault(module_name, [])
        if lst and self.__average_factors:
            lst[0].add_(inputs_cov)
        else:
            lst.append(inputs_cov)

        # Assume this is the last layer until fwd-ing through another linear
        self.__last_linear = layer_idx
        batch_size, out_no = output.size()
        self.__expected_output_hessian_size = torch.Size([batch_size, out_no, out_no])

    def __linear_bwd_hook(self, module, grad_input, grad_output):
        assert self.__phase == self.BACKWARD
        module_name = self.__module_names[id(module)]
        if self.__maybe_exact:
            hess = self.__linear_exact_hessian(module, grad_input, grad_output)
        else:
            hess = self.__linear_fisher(module, grad_input, grad_output)

        lst = self.__outputs_hess.setdefault(module_name, [])
        if lst and self.__average_factors:
            lst[0].add_(hess)
        else:
            lst.append(hess)

    def __linear_fisher(self, _module, _grad_input, grad_output):
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        dy, = grad_output
        return (dy.t() @ dy).div_(dy.size(0)).detach_()

    def __linear_exact_hessian(self, _module, grad_input, grad_output):
        assert isinstance(grad_input, tuple) and len(grad_input) == 3
        assert isinstance(grad_output, tuple) and len(grad_output) == 1

        dx, dw, db = grad_input
        dz, = grad_output

        layer_idx = self.__layer_idx
        if layer_idx == self.__last_linear:
            assert self.__next_parametric is None
            if self.__last_hessian is None:
                # Compute Fisher
                act = torch.bmm(dz.unsqueeze(2), dz.unsqueeze(1)).detach_()
            else:
                act = self.__output_hessian.detach_()
        else:
            next_dz = self.__next_outputs_hessian
            next_w = self.__next_parametric.weight
            df_dx = self.__df_dx[layer_idx + 1]

            b_sz, next_out_no, _ = next_dz.size()
            _next_out_no, crt_out_no = next_w.size()
            assert next_out_no == _next_out_no
            assert df_dx.size() == torch.Size([b_sz, crt_out_no])

            left_diag = df_dx.unsqueeze(2).expand(b_sz, crt_out_no, next_out_no)
            left_w = next_w.t().unsqueeze(0).expand(b_sz, crt_out_no, next_out_no)
            left = torch.mul(left_diag, left_w)
            act = torch.matmul(left, next_dz)
            right = left.transpose(1, 2)
            act = torch.matmul(act, right)
            snd_order = self.__d2f_dx2[layer_idx + 1]
            for i in range(act.size(0)):
                act[i].add_(torch.diag(snd_order[i]))
            act.detach_()

            del self.__df_dx[layer_idx + 1]
            del self.__d2f_dx2[layer_idx + 1]

        self.__next_outputs_hessian = act
        return act.mean(0).detach_()

    def __activation_fwd_hook(self, _module, inputs, output):
        layer_idx = self.__layer_idx
        inputs, = inputs
        df_dx, = autograd.grad(output, inputs,
                               grad_outputs=torch.ones_like(inputs),
                               create_graph=True, retain_graph=True)
        d2f_dx2, = autograd.grad(df_dx, inputs,
                                 grad_outputs=torch.ones_like(df_dx),
                                 create_graph=True, retain_graph=False)
        self.__df_dx[layer_idx] = df_dx.detach()
        self.__d2f_dx2[layer_idx] = d2f_dx2.detach()

    def __activation_bwd_hook(self, _module, _grad_input, grad_output):
        layer_idx = self.__layer_idx
        assert layer_idx in self.__df_dx and layer_idx in self.__d2f_dx2
        grad_output, = grad_output
        self.__d2f_dx2[layer_idx] *= grad_output.detach()

    def __conv_fwd_hook(self, module, inputs, output):
        module_name = self.__module_names[id(module)]
        assert isinstance(inputs, tuple) and len(inputs) == 1
        assert isinstance(output, Tensor)
        inputs, = inputs

        ch_out, ch_in, k_h, k_w = module.weight.size()
        s_h, s_w = module.stride
        b_sz, ch_in_, h_in, w_in = inputs.size()
        h_out = (h_in - k_h + 0) // s_h + 1
        w_out = (w_in - k_w + 0) // s_w + 1
        b_sz_, ch_out_, h_out_, w_out_ = output.size()

        assert ch_in_ == ch_in
        assert h_out_ == h_out
        assert w_out == w_out_ and \
            ch_out_ == ch_out and b_sz_ == b_sz

        # TODO: search for faster ways to do this
        x = inputs.new().resize_(b_sz, h_out, w_out, ch_in, k_h, k_w)
        for idx_h in range(0, h_out):
            start_h = idx_h * s_h
            for idx_w in range(0, w_out):
                start_w = idx_w * s_w
                x[:, idx_h, idx_w, :, :, :].copy_(
                    inputs[:, :, start_h:(start_h + k_h), start_w:(start_w + k_w)]
                )

        x = x.view(b_sz * h_out * w_out, ch_in * k_h * k_w)
        if self.__do_checks:
            # Keep them until bwd pass
            self.__conv_special_inputs[module_name] = x

        x = torch.cat([x, x.new_ones(b_sz * h_out * w_out, 1)], dim=1)

        if self.__do_checks:
            weight_extra = torch.cat([module.weight.view(ch_out, -1),
                                      module.bias.view(ch_out, -1)], dim=1)
            y = (x @ weight_extra.t()).view(b_sz, h_out * w_out, ch_out)\
                                      .transpose(1, 2)\
                                      .view(b_sz, ch_out, h_out, w_out)
            assert (y - output).abs().max() < 1e-5  # assert torch.allclose(y, output)

        inputs_cov = (x.t() @ x).div_(b_sz)

        lst = self.__inputs_cov.setdefault(module_name, [])
        if lst and self.__average_factors:
            lst[0].add_(inputs_cov)
        else:
            lst.append(inputs_cov)

    def __conv_bwd_hook(self, module, grad_input, grad_output):
        assert isinstance(grad_input, tuple) and len(grad_input) == 3
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        module_name = self.__module_names[id(module)]
        dx, dw, db = grad_input
        dy, = grad_output
        b_sz, ch_out, h_out, w_out = dy.size()
        dy = dy.view(b_sz, ch_out, -1).transpose(1, 2)\
                                      .contiguous().view(-1, ch_out)

        if self.__do_checks:
            ch_out_, ch_in, _k_h, _k_w = module.weight.size()
            assert ch_out == ch_out_
            x = self.__conv_special_inputs[module_name]
            b_sz = dx.size(0)
            ch_out = dy.size(1)
            dw_ = torch.mm(dy.t(), x).view_as(dw)
            assert (dw - dw_).sum().item() < 1e-5

        act = (dy.t() @ dy).div_(b_sz * h_out * w_out).detach_()

        lst = self.__outputs_hess.setdefault(module_name, [])
        if lst and self.__average_factors:
            lst[0].add_(act)
        else:
            lst.append(act)

    def end_kf(self):
        if not self.__kf_mode:
            raise Exception("You forgot to activate KF mode.")
        if self.__average_factors:
            coeff = 1.0 / float(self.__batches_no)
            for tensors in self.__inputs_cov.values():
                assert len(tensors) == 1
                tensors[0].mul_(coeff)
            for tensors in self.__outputs_hess.values():
                assert len(tensors) == 1
                tensors[0].mul_(coeff)
        kfhp = KFHessianProduct(self.__inputs_cov, self.__outputs_hess)
        self.do_kf = False  # Do not put above as this drops tensors
        return kfhp
