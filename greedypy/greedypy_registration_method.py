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
        outdir,
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

        None
