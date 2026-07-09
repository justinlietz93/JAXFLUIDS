from typing import List, Tuple, Protocol
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids import InputManager
from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.initialization.helper_functions import create_field_buffer


Array = jax.Array

def reassemble_buffer(
        buffer: np.ndarray,
        split_factors: Tuple,
        jax_numpy = False,
        keep_transpose = False,
    )-> np.ndarray:
    """Reassembles a decomposed buffer.
    jax_numpy specifies whether jax or
    numpy operators are used.

    :param buffer: Buffer with shape (Ni,Nz+2*Nh,Ny+2*Nh,Nx+2*Nh,...)
        where Ni is the number of subdomains, Nx, Ny, Nz, are
        the number of cells, and Nh are the number of halo
        cells
    :type buffer: Array
    :param split_factors: Specifies the domain decomposition, defaults to None
    :type split_factors: Tuple, optional
    :param nh: Number of halo cells, defaults to None
    :type nh: int, optional
    :return: _description_
    :rtype: Array
    """
    if jax_numpy:
        return reassemble_buffer_jnp(buffer, split_factors, keep_transpose)
    else:
        return reassemble_buffer_np(buffer, split_factors, keep_transpose)

@partial(jax.jit, static_argnums=(1,2))
def reassemble_buffer_jnp(
        buffer: Array,
        split_factors: Tuple,
        keep_transpose: bool,
    ) -> Array:
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = jnp.reshape(buffer, reshape)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
    if not keep_transpose:
        buffer = jnp.transpose(buffer)
    return buffer

def reassemble_buffer_np(
        buffer: np.ndarray,
        split_factors: Tuple,
        keep_transpose: bool
        ) -> np.ndarray:
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = np.reshape(buffer, reshape)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
    if not keep_transpose:
        buffer = np.transpose(buffer)
    return buffer

def split_subdomain_dimensions(
    buffer: np.ndarray,
    split_factors: Tuple
    ) -> np.ndarray:
    """Splits up the subdomain dimensions of the buffer.

    :param buffer: _description_
    :type buffer: np.ndarray
    :return: _description_
    :rtype: np.ndarray
    """
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = buffer.reshape(reshape)
    return buffer

def flatten_subdomain_dimensions(buffer: np.ndarray) -> np.ndarray:
    """Flattens the subdomain dimensions of the buffer.
    """
    shape = buffer.shape
    reshape = (-1,) + shape[3:]
    buffer = buffer.reshape(reshape)
    return buffer


class FillHalosFn(Protocol):
    def __call__(
        self,
        primitives: Array,
        physical_simulation_time: float,
        fill_edge_halos: bool = True,
        fill_vertex_halos: bool = True,
        conservatives: Array | None = None,
        fill_face_halos: bool = True,
        ml_setup: MachineLearningSetup | None = None,
    ) -> Array | tuple[Array, Array]:
        ...


def make_fill_halos_fn(
        case_setup: str,
        numerical_setup: str,
    ) -> FillHalosFn:
    """Creates a postprocessing function for
    filling halo cells.
    """

    input_manager = InputManager(case_setup, numerical_setup)
    halo_manager = input_manager.halo_manager

    domain_information = input_manager.domain_information

    if domain_information.is_parallel:
        raise NotImplementedError(
            "fill_halos_fn currently supports only unsplit postprocessing arrays. "
            "Use a case setup with split_x = split_y = split_z = 1, or gather the "
            "parallel output to a global array before calling this helper."
        )

    nhx, nhy, nhz = domain_information.domain_slices_conservatives
    device_number_of_cells = domain_information.device_number_of_cells
    nh = domain_information.nh_conservatives

    equation_information = input_manager.equation_information
    equation_type = equation_information.equation_type
    no_primes = equation_information.no_primes
    leading_dim = (5,2) if equation_type == "TWO-PHASE-LS" else no_primes

   
    def fill_halos_fn(
        primitives: Array,
        physical_simulation_time: float,
        fill_edge_halos: bool = True,
        fill_vertex_halos: bool = True,
        conservatives: Array | None = None,
        fill_face_halos: bool = True,
        ml_setup: MachineLearningSetup | None = None,
    ) -> Array | tuple[Array, Array]:
        """Fill conservative-domain halo cells for material fields.

        Parameters
        ----------
        primitives:
            Primitive variables without halo cells, shaped like
            ``(..., nx, ny, nz)`` for the unsplit device domain.
        physical_simulation_time:
            Physical simulation time passed to boundary-condition callables.
        fill_edge_halos:
            Whether to fill edge halo cells.
        fill_vertex_halos:
            Whether to fill vertex halo cells.
        conservatives:
            Optional conservative variables without halo cells. When provided,
            conservative halo cells are filled consistently with ``primitives``.
        fill_face_halos:
            Whether to fill face halo cells.
        ml_setup:
            Optional machine-learning setup forwarded to the JAX-Fluids halo manager.

        Returns
        -------
        Array or tuple[Array, Array]
            Primitive buffer with halo cells if ``conservatives`` is ``None``;
            otherwise the halo-filled primitive and conservative buffers.

        Notes
        -----
        This helper currently assumes an unsplit case setup
        ``split_x = split_y = split_z = 1``.
        """
        
        if ml_setup is None:
            ml_setup = MachineLearningSetup()

        buffer = create_field_buffer(
            nh,
            device_number_of_cells,
            primitives.dtype,
            leading_dim,
        )
        primitives = buffer.at[...,nhx,nhy,nhz].set(primitives)

        if conservatives is not None:
            buffer = create_field_buffer(
                nh,
                device_number_of_cells,
                conservatives.dtype,
                leading_dim,
            )
            conservatives = buffer.at[...,nhx,nhy,nhz].set(conservatives)

        return halo_manager.perform_halo_update_material(
            primitives,
            physical_simulation_time,
            fill_edge_halos,
            fill_vertex_halos,
            conservatives,
            fill_face_halos,
            ml_setup,
        )

    return fill_halos_fn