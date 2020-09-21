import numpy as np
from scipy.ndimage import zoom
import greedypy.metrics as metrics
import greedypy.regularizers as regularizers
import greedypy.transformer as transformer



class greedypy_registration_method:
    """
    """

    def __init__(
        self,
        fixed, fixed_vox,
        moving, moving_vox,
        iterations,
        shrink_factors,
        smooth_sigmas,
        radius=16,
        early_convergence_ratio=1e-4,
        convergence_test_length=10,
        field_abcd=[0.5, 0., 1., 6.],
        gradient_abcd=[3., 0., 1., 2.],
        dtype=np.float32,
        step=None,
        log=None,
    ):
        """
        """

        error = "iterations, shrink_factors, and smooth_sigmas must be equal length lists"
        assert (len(iterations) == len(shrink_factors) and 
                len(iterations) == len(smooth_sigmas)
        ), error

        if step is None:
             step = fixed_vox.min()
        self.__dict__.update(locals())
        self.phi = None
        self.warped = None
        self.invphi = None
        self.mask = None
        self.initial_transform = None


    def set_mask(self, mask):
        """
        """

        self.mask = mask


    def mask_values(self, values):
        """
        """

        if self.mask is None:
            self.mask = np.ones_like(self.moving)
        if type(values) is not list:
            values = [values,]
        for value in values:
            self.mask[self.moving == value] = 0


    def set_initial_transform(self, initial_transform):
        """
        """

        self.initial_transform = initial_transform


    def get_warp(self):
        """
        """

        return self.phi


    def get_inverse_warp(self):
        """
        """

        trans = transformer.transformer(
            self.fixed.shape, self.fixed_vox, dtype=self.dtype,
        )
        self.invphi = trans.invert(self.phi)
        return self.invphi


    def get_warped_image(self):
        """
        """

        return self.warped


    def optimize(self):
        """
        """

        # loop over resolution levels
        for level, local_iterations in enumerate(self.iterations):

            # resample images
            fixed = self._downsample(
                self.fixed,
                self.fixed_vox,
                1./self.shrink_factors[level],
                alpha=self.smooth_sigmas[level],
            )
            moving = self._downsample(
                self.moving,
                self.moving_vox,
                1./self.shrink_factors[level],
                alpha=self.smooth_sigmas[level],
            )

            # new voxel sizes and step
            fixed_vox = self.fixed_vox * self.shrink_factors[level]
            moving_vox = self.moving_vox * self.shrink_factors[level]
            step = self.step * self.shrink_factors[level]

            # initialize or resample the transform
            if self.phi is None:
                phi = np.zeros(
                    fixed.shape + (len(fixed.shape),),
                    dtype=self.dtype,
                )
            else:
                zoom_factor = np.array(fixed.shape) / np.array(self.phi.shape[:-1])
                phi = [zoom(self.phi[..., i], zoom_factor, order=3, mode='nearest')
                    for i in range(3)
                ]
                phi = np.ascontiguousarray(np.moveaxis(np.array(phi), 0, -1))
            self.phi = phi

            # resample the residual mask
            if self.mask is not None:
                mask = self._downsample(
                    self.mask,
                    self.fixed_vox,
                    1./self.shrink_factors[level],
                    alpha=self.smooth_sigmas[level],
                    order=0,
                )

            # transformer
            trans = transformer.transformer(
                fixed.shape, fixed_vox,
                initial_transform=self.initial_transform,
                dtype=self.dtype
            )

            # smoothers
            field_smoother = regularizers.differential(
                self.field_abcd[0] * self.shrink_factors[level],
                *self.field_abcd[1:], fixed_vox, fixed.shape, self.dtype,
            )
            grad_smoother = regularizers.differential(
                self.gradient_abcd[0] * self.shrink_factors[level],
                *self.gradient_abcd[1:], fixed_vox, fixed.shape, self.dtype,
            )

            # metric
            metric = metrics.local_correlation(fixed, moving, self.radius)

            # optimization variables
            iteration = 0
            converged = False
            energy_history = []

            # optimize at this level
            while iteration < local_iterations and not converged:

                # compute the residual
                warped = trans.apply_transform(moving, phi)
                energy, gradient = metric.gradient(fixed, warped, self.radius, fixed_vox)
                gradient = grad_smoother.smooth(gradient)

                # apply moving image mask to residual
                if self.mask is not None:
                    gradient = gradient * mask[..., None]

                # monitor the optimization
                if iteration == 0:
                     initial_energy = energy
                if iteration < self.convergence_test_length:
                    energy_history.append(energy)
                else:
                    energy_history.pop(0)
                    energy_history.append(energy)
                    x = np.gradient(energy_history).mean()
                    y = initial_energy - energy_history[-1]
                    if x > 0 or abs( x/y ) < self.early_convergence_ratio:
                        converged = True

                # make the gradient descent update
                scale = step / np.linalg.norm(gradient, axis=-1).max()
                phi = phi - scale * gradient
                phi = field_smoother.smooth(phi)

                # record progress
                self._record("Level: {}, Iteration: {}, Energy: {}".format(level, iteration, energy))

                # the wheel keeps on spinning
                iteration = iteration + 1

            # store the transform for the next level
            self.phi = phi
            self.warped = warped


    def _downsample(self, image, spacing, zoom_factor, alpha=1., order=1):
        """
        """

        if zoom_factor > 1.:
            raise ValueError('zoom_factor must be less than 1 for _downsample')
        if zoom_factor == 1. and alpha == 0.:
            return image

        smoother = regularizers.differential(
            alpha, 0., 1., 2.,
            spacing,
            image.shape,
            dtype=self.dtype,
        )
        return zoom(
            smoother.smooth(image),
            zoom_factor, mode='reflect', order=order,
        )


    def _record(self, message):
        """
        """

        print(message)
        if self.log is not None:
            print(message, file=self.log)


