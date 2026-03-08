import numpy as np

from comp_kraus import comp_kraus
from run_opt_full import run_opt_full


def gadc_kraus(gamma, population):
    root_one_minus_population = np.sqrt(1 - population)
    root_population = np.sqrt(population)

    K0 = root_one_minus_population * np.array(
        [[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128
    )
    K1 = np.sqrt(gamma * (1 - population)) * np.array(
        [[0, 1], [0, 0]], dtype=np.complex128
    )
    K2 = root_population * np.array(
        [[np.sqrt(1 - gamma), 0], [0, 1]], dtype=np.complex128
    )
    K3 = np.sqrt(gamma * population) * np.array(
        [[0, 0], [1, 0]], dtype=np.complex128
    )
    return [K0, K1, K2, K3]


def damp_dephasing(p_channel, g_channel):
    s = np.sqrt(1.0 - g_channel)

    K0 = np.sqrt(1.0 - p_channel) * np.array(
        [[1.0, 0.0],
         [0.0, s]], dtype=np.complex128
    )
    K1 = np.sqrt(g_channel) * np.array(
        [[0.0, 1.0],
         [0.0, 0.0]], dtype=np.complex128
    )
    K2 = np.sqrt(p_channel) * np.array(
        [[1.0, 0.0],
        [0.0, -s]], dtype=np.complex128
    )

    return [K0, K1, K2]


def example_full():
    gamma = 0.2
    p_channel = 0.16

    # K = gadc_kraus(gamma, population)
    K = damp_dephasing(p_channel=p_channel, g_channel=gamma)
    Kc = comp_kraus(K, dims=(2, 2))

    n = 3
    k = 2

    print("\n--- Running search (Stiefel x Simplex) for n=%d, k=%d ---" % (n, k))
    Ic_full, rho_full, lambda_full = run_opt_full(K, Kc, n, k,verbose=True)

    print("Full search results (optimized lambda): %.8g" % (Ic_full / n))
    print("Optimal eigenvalues Found:", lambda_full)
    print("\nOptimal state from full search (rho_full):")
    print(rho_full)


if __name__ == "__main__":
    example_full()
