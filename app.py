import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Метод вращений", layout="wide")

# Функция для проверки симметричности матрицы
def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

# Функция для нахождения максимального внедиагонального элемента
def find_max_offdiag(A):
    n = A.shape[0]
    max_val = 0
    p, q = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return p, q, max_val

# Функция для вычисления угла вращения
def compute_rotation(A, p, q):
    if abs(A[p, q]) < 1e-20:
        return 1.0, 0.0
    
    if abs(A[p, p] - A[q, q]) < 1e-12:
        theta = np.pi/4 if A[p, q] > 0 else -np.pi/4
    else:
        theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])
    
    c = np.cos(theta)
    s = np.sin(theta)
    return c, s

# Функция для выполнения вращения
def rotate(A, V, p, q, c, s):
    n = A.shape[0]
    A_new = A.copy()
    V_new = V.copy()
    
    # Обновление строк и столбцов p и q
    for i in range(n):
        if i != p and i != q:
            # Обновление элементов матрицы A
            a_ip = A[i, p]
            a_iq = A[i, q]
            A_new[i, p] = A_new[p, i] = c * a_ip - s * a_iq
            A_new[i, q] = A_new[q, i] = s * a_ip + c * a_iq
    
    # Обновление элементов на пересечении p и q
    a_pp = A[p, p]
    a_qq = A[q, q]
    a_pq = A[p, q]
    
    A_new[p, p] = c*c*a_pp - 2*c*s*a_pq + s*s*a_qq
    A_new[q, q] = s*s*a_pp + 2*c*s*a_pq + c*c*a_qq
    A_new[p, q] = A_new[q, p] = 0.0
    
    # Обновление матрицы собственных векторов
    for i in range(n):
        v_ip = V[i, p]
        v_iq = V[i, q]
        V_new[i, p] = c * v_ip - s * v_iq
        V_new[i, q] = s * v_ip + c * v_iq
    
    return A_new, V_new

# Основная функция метода вращений
def jacobi_method(A, eps=1e-8, max_iter=1000):
    n = A.shape[0]
    A_current = A.copy()
    V = np.eye(n)
    
    iter_count = 0
    off_diag_norms = []
    
    while iter_count < max_iter:
        p, q, max_offdiag = find_max_offdiag(A_current)
        
        # Норма внедиагональных элементов для критерия остановки
        off_diag_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                off_diag_sum += A_current[i, j]**2
        off_diag_norm = np.sqrt(2 * off_diag_sum)
        off_diag_norms.append(off_diag_norm)
        
        if max_offdiag < eps:
            break
        
        # Вычисление угла вращения
        c, s = compute_rotation(A_current, p, q)
        
        # Применение вращения
        A_current, V = rotate(A_current, V, p, q, c, s)
        
        iter_count += 1
    
    eigenvalues = np.diag(A_current)
    
    # Сортировка собственных значений и векторов
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    return eigenvalues, V, iter_count, off_diag_norms

# Функция для проверки решения
def verify_solution(A, eigenvalues, eigenvectors):
    n = len(eigenvalues)
    residuals = []
    for i in range(n):
        v = eigenvectors[:, i]
        Av = A @ v
        lambda_v = eigenvalues[i] * v
        residual = np.linalg.norm(Av - lambda_v)
        residuals.append(residual)
    max_residual = max(residuals)
    return max_residual, residuals

# Функция для исследования сходимости
def study_convergence(A, epsilons):
    results = []
    for eps in epsilons:
        start_time = time.time()
        _, _, iterations, _ = jacobi_method(A.copy(), eps=eps, max_iter=10000)
        end_time = time.time()
        results.append((eps, iterations, end_time - start_time))
    return results

# Заголовок приложения
st.title("Метод вращений для нахождения собственных значений и векторов")
st.markdown("""
Это приложение реализует метод вращений (метод Якоби) для нахождения собственных значений и векторов симметричных матриц.
""")

# Боковая панель для настроек
with st.sidebar:
    st.header("Параметры")
    n = st.slider("Размерность матрицы", min_value=2, max_value=8, value=3, 
                 help="Выберите размерность квадратной матрицы")
    eps = st.number_input("Точность (ε)", min_value=1e-12, max_value=1e-1, value=1e-6, 
                         format="%.1e", step=1e-7,
                         help="Критерий остановки: вычисления прекращаются, когда максимальный внедиагональный элемент становится меньше ε")
    max_iter = st.number_input("Макс. число итераций", min_value=10, max_value=10000, value=1000,
                              help="Предельное число итераций для избежания зацикливания")

# Ввод матрицы
st.header("Введите симметричную матрицу")

# Создание пустой матрицы
matrix_input = np.zeros((n, n))

# Интерфейс для ввода матрицы
for i in range(n):
    cols = st.columns(n)
    for j in range(n):
        if j >= i:  # Позволяем вводить только верхний треугольник
            # Значение по умолчанию для диагональных элементов - 1, для остальных - 0
            default_val = 1.0 if i == j else 0.0
            val = cols[j].number_input(f"a{i+1}{j+1}", value=default_val, key=f"{i}_{j}")
            matrix_input[i, j] = val
            matrix_input[j, i] = val  # Обеспечиваем симметричность
        else:
            # Отображаем зеркальные значения серым цветом
            cols[j].text_input(f"a{i+1}{j+1}", value=f"{matrix_input[i, j]:.4f}", disabled=True, key=f"disabled_{i}_{j}")

# Отображение текущей матрицы
st.subheader("Текущая матрица:")
st.dataframe(pd.DataFrame(matrix_input))

# Проверка симметричности
if not is_symmetric(matrix_input):
    st.error("⚠️ **Внимание:** Матрица не является симметричной. Метод вращений применим только для симметричных матриц.", icon="⚠️")
else:
    st.success("✅ **Матрица симметрична.** Метод вращений может быть применен.", icon="✅")

    # Кнопка для запуска вычислений
    if st.button("🚀 Запустить вычисления", type="primary", use_container_width=True):
        # Выполнение метода
        start_time = time.time()
        eigenvalues, eigenvectors, iterations, norms = jacobi_method(
            matrix_input.copy(), eps=eps, max_iter=max_iter)
        end_time = time.time()
        
        # Отображение результатов
        st.header("Результаты вычислений")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Время вычислений", f"{end_time - start_time:.6f} секунд")
        with col2:
            st.metric("Число итераций", iterations)
        
        # Собственные значения
        st.subheader("Собственные значения:")
        eigenvalues_df = pd.DataFrame({
            "№": range(1, len(eigenvalues)+1),
            "Значение": eigenvalues
        })
        st.dataframe(eigenvalues_df)
        
        # Собственные векторы
        st.subheader("Собственные векторы:")
        st.markdown("Каждый столбец матрицы ниже - это собственный вектор, соответствующий собственному значению")
        eigenvectors_df = pd.DataFrame(eigenvectors)
        eigenvectors_df.columns = [f"Вектор {i+1}" for i in range(eigenvectors.shape[1])]
        eigenvectors_df.index = [f"x{i+1}" for i in range(eigenvectors.shape[0])]
        st.dataframe(eigenvectors_df)
        
        # Проверка решения
        st.subheader("Проверка решения")
        max_residual, residuals = verify_solution(matrix_input, eigenvalues, eigenvectors)
        
        if max_residual < 1e-6:
            st.success(f"✅ **Решение верное!** Максимальная невязка: {max_residual:.2e}", icon="✅")
        else:
            st.warning(f"⚠️ **Предупреждение:** Невязка велика. Максимальная невязка: {max_residual:.2e}", icon="⚠️")
        
        # Таблица невязок для каждого вектора
        residuals_df = pd.DataFrame({
            "Вектор №": range(1, len(residuals)+1),
            "Невязка": residuals
        })
        st.dataframe(residuals_df)
        
        # График сходимости
        st.subheader("График сходимости")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(range(len(norms)), norms, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel("Номер итерации", fontsize=12)
        ax.set_ylabel("Норма внедиагональных элементов", fontsize=12)
        ax.grid(True, which="both", ls="-")
        ax.set_title("Сходимость метода вращений", fontsize=14)
        st.pyplot(fig)
        
        # Исследование сходимости
        if st.checkbox("📊 Исследовать зависимость сходимости от точности", value=False):
            epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            convergence_results = study_convergence(matrix_input.copy(), epsilons)
            
            st.subheader("Зависимость числа итераций и времени от точности")
            
            # Подготовка данных для графиков
            eps_values = [res[0] for res in convergence_results]
            iter_values = [res[1] for res in convergence_results]
            time_values = [res[2] for res in convergence_results]
            
            # Создание двух графиков рядом
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # График зависимости числа итераций от точности
            ax1.loglog(eps_values, iter_values, 'ro-', linewidth=2, markersize=8)
            ax1.set_xlabel("Точность (ε)", fontsize=12)
            ax1.set_ylabel("Число итераций", fontsize=12)
            ax1.grid(True, which="both", ls="-")
            ax1.set_title("Зависимость числа итераций от точности", fontsize=14)
            
            # График зависимости времени вычислений от точности
            ax2.loglog(eps_values, time_values, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel("Точность (ε)", fontsize=12)
            ax2.set_ylabel("Время вычислений (сек)", fontsize=12)
            ax2.grid(True, which="both", ls="-")
            ax2.set_title("Зависимость времени вычислений от точности", fontsize=14)
            
            st.pyplot(fig2)
            
            # Таблица результатов
            st.subheader("Таблица результатов исследования")
            results_df = pd.DataFrame(convergence_results, columns=["Точность (ε)", "Число итераций", "Время (сек)"])
            results_df["Точность (ε)"] = results_df["Точность (ε)"].apply(lambda x: f"{x:.0e}")
            results_df["Время (сек)"] = results_df["Время (сек)"].apply(lambda x: f"{x:.6f}")
            st.dataframe(results_df)

# Справочная информация
with st.sidebar:
    st.markdown("---")
    st.subheader("О методе вращений")
    st.markdown("""
    **Метод вращений (метод Якоби)** - итерационный метод нахождения собственных значений и векторов симметричных матриц.
    
    **Основная идея:** последовательное обнуление внедиагональных элементов матрицы с помощью ортогональных преобразований вращения.
    
    **Преимущества метода:**
    - Простота реализации
    - Гарантированная сходимость для симметричных матриц
    - Устойчивость к ошибкам округления
    
    **Критерий остановки:** когда максимальный по модулю внедиагональный элемент становится меньше заданной точности ε.
    """)
    
    st.markdown("**Важно:** Метод применим только для симметричных матриц, т.е. матриц, для которых A = Aᵀ.")

