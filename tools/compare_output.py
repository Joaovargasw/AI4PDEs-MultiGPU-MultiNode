import numpy as np
import matplotlib.pyplot as plt

def compare_results(file_z, file_quad, tolerance=1e-5):
    # Assumindo que você salvou os dados como .npy ou binário
    data_z = np.load(file_z)
    data_quad = np.load(file_quad)

    # 1. Verificação de Shape
    if data_z.shape != data_quad.shape:
        print(f"ERRO: Formatos diferentes! Z: {data_z.shape}, Quad: {data_quad.shape}")
        return

    # 2. Mapa de Diferença Absoluta
    diff = np.abs(data_z - data_quad)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # 3. Erro Relativo (L2 Norm) - Melhor métrica global
    # Evita divisão por zero adicionando um epsilon pequeno
    norm_diff = np.linalg.norm(data_z - data_quad)
    norm_ref = np.linalg.norm(data_z)
    relative_error = norm_diff / (norm_ref + 1e-10)

    print(f"--- Relatório de Comparação ---")
    print(f"Máxima Diferença Absoluta: {max_diff:.8e}")
    print(f"Erro Médio: {mean_diff:.8e}")
    print(f"Erro Relativo (L2): {relative_error:.8e}")

    is_similar = np.allclose(data_z, data_quad, rtol=tolerance, atol=tolerance)
    print(f"Passou na tolerância ({tolerance})? {'SIM' if is_similar else 'NÃO'}")

    # 4. Onde está o erro? (Visualização)
    # Plota a fatia central para ver se o erro está nas bordas
    mid_idx = data_z.shape[0] // 2
    plt.imshow(diff[mid_idx, :, :], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Diferença Absoluta')
    plt.title(f"Mapa de Erro (Fatia {mid_idx})")
    plt.show()

# Exemplo de uso
# compare_results('resultado_z.npy', 'resultado_quadrantes.npy')
