import numpy as np
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
        radius=8,
        step=1e-2,
        tolerance=1e-2,
        field_abcd=[3., 0., 1., 2.],
        gradient_abcd=[3., 0., 1., 2.],
        dtype=np.float32,
        log=None,
    ):
        """
        """

        self.__dict__.update(locals())
        self.grid = fixed.shape
        self.mask = None
        self.auto_mask = None
        self.initial_transform = None


    def set_mask(self, mask):
        """
        """

        self.mask = mask


    def set_auto_mask(self, auto_mask):
        """
        """

        self.auto_mask = auto_mask


    def set_initial_transform(self, initial_transform):
        """
        """

        self.initial_transform = initial_transform


    def save_warped_image(self):
        """
        """

        None


    def save_final_lcc(self):
        """
        """

        None


    def compose_result_and_initial_transform(self):
        """
        """

        None


    def inverse_result(self):
        """
        """

        None


    def optimize(self):
        """
        """

        self.phi = None  # the optimal transform
        for level, local_iterations in enumerate(self.iterations):

            # resample images
            level_countdown = len(self.iterations) - level - 1
            fixed = self._resample(
                self.fixed,
                self.fixed_vox,
                1./2**level_countdown,
                alpha=1.+level_countdown,
            )
            moving = self._resample(
                self.moving,
                self.moving_vox,
                1./2**level_countdown,
                alpha=1.+level_countdown,
            )

            # new voxel sizes
            fixed_vox = self.fixed_vox * 2**level_countdown
            moving_vox = self.moving_vox * 2**level_countdown

            # initialize or resample the transform
            if self.phi is None:
                phi = np.zeros(
                    fixed.shape + (len(fixed.shape),),
                    dtype=self.dtype,
                )
            else:
                zoom_factor = np.array(fixed.shape) / np.array(self.phi.shape[:-1])
                # TODO: nearest seems wrong? check docs and test with linear
                phi = [zoom(self.phi[..., i], zoom_factor, mode='nearest')
                    for i in range(3)
                ]
                phi = np.ascontiguousarray(np.moveaxis(np.array(phi), 0, -1))
            self.phi = phi

            # initialize the transformer
            trans = transformer.transformer(
                fixed.shape, fixed_vox, self.dtype
            )
            if self.initial_transform is not None:
                trans.set_initial_moving_transform(self.initial_transform)

            # initialize the smoothers
            field_smoother = regularizers.differential(
                self.field_abcd[0] * 2**level_countdown,
                *self.field_abcd[1:], fixed_vox, fixed.shape, self.dtype,
            )
            grad_smoother = regularizers.differential(
                self.grad_abcd[0] * 2**level_countdown,
                *self.grad_abcd[1:], fixed_vox, fixed.shape, self.dtype,
            )

            # initialize the metric
            metric = metrics.local_correlation(fixed, moving, self.lcc_radius)

            # initialize the mask
            if self.auto_mask is not None:
                mask = np.ones(moving.shape, dtype=np.uint8)
                # TODO: apply initial transform to moving image, call it transformed
                for intensity in self.auto_mask:
                    mask[transformed==intensity] = 0

            # some flags and values to keep track of
            iteration = 0  # the iteration counter
            backstep_count = 0  # number of times the optimization has reversed a step
            converged = False  # flag for early convergence
            self.energy = 0  # the lowest energy

            # optimize at this level
            while iteration < local_iterations and not converged:

                # compute residual
                # apply moving image mask to residual
                # monitor the optimization
                # make the gradient descent update
                # record progress


    def _downsample(image, spacing, zoom_factor, alpha=1.):
        """
        """

        smoother = regularizers.differential(
            alpha, 0., 1., 2.,
            spacing,
            image.shape,
            dtype=self.dtype,
        )
        # TODO: test if copy is necessary here  
        return zoom(
            np.copy(smoother.smooth(image)),
            zoom_factor, mode='wrap',
        )


    def _record(message, log):
        """
        """

        print(message)
        if self.log is not None:
            print(message, file=log)
