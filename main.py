# -------------------- Librerías --------------------------------
from sympy import symbols, Eq, solve, cos, sin, pi, sqrt, Rational
from colorama import init, Fore
import numpy as np
import matplotlib.pyplot as plt
# -------------------- Interaccion de Usuario -------------------
# Se ingresa la cantidad de vectores
def get_vectors():
    num_vectors = int(input("Cuantos vectores son? "))
    vectors = []
    for i in range(num_vectors):
        vector = input(f"Ingresa el vector {chr(65 + i)} (ej: 1,2): ")
        vector = tuple(map(float, vector.split(',')))
        vectors.append(vector)

    return vectors

# -------------------- Procedimientos matematicos --------------------
# Calculo de ecuaciones para la base
def generar_ecuaciones(vectores):
    # Crear símbolos (a, b, c, ...)
    variables = symbols(f"a:{len(vectores)}")
    
    # Crear ecuaciones de la forma: a * vector[0] + b * vector[1] + ... = 0
    ecuaciones = []
    for i in range(len(vectores[0])):  # Iterar por la dimensión del vector
        suma = sum(variables[j] * vectores[j][i] for j in range(len(vectores)))
        ecuaciones.append(Eq(suma, 0))  # Cada fila suma a 0

    return ecuaciones, variables

def formatear_termino(termino, decimales=2):
    coef, var = termino.as_coeff_Mul()  # Separa el coeficiente y la variable
    coef = round(float(coef), decimales)  # Redondear coeficiente y asegurarse de que sea flotante
    if coef == 1:  # Si el coeficiente es 1, deja solo la variable
        return str(var)
    elif coef == -1:  # Si el coeficiente es -1, agrega solo el signo negativo y la variable
        return f"-{var}"
    elif coef == 0:
        return ""
    else:  # Para otros coeficientes, combina el coeficiente y la variable
        return f"{coef}{var}"

# Calculo del determinante
def determinante(matriz):
    # Verificar si la matriz es cuadrada
    if not all(len(fila) == len(matriz) for fila in matriz):
        raise ValueError("La matriz debe ser cuadrada.")
    
    # Caso base: determinante de una matriz 1x1
    if len(matriz) == 1:
        return matriz[0][0]
    
    # Caso base: determinante de una matriz 2x2
    if len(matriz) == 2:
        return matriz[0][0] * matriz[1][1] - matriz[0][1] * matriz[1][0]
    
    # Caso general: cálculo del determinante por cofactores
    det = 0
    for columna, valor in enumerate(matriz[0]):
        # Generar la submatriz excluyendo la fila 0 y la columna actual
        submatriz = tuple(
            tuple(fila[:columna] + fila[columna + 1:])
            for fila in matriz[1:]
        )
        # Recursión para el determinante de la submatriz
        det += ((-1) ** columna) * valor * determinante(submatriz)
    
    return det

def calcularBase(vectores):
    # Calcular el determinante
    determinanteActual = determinante(vectores)
    # Generar las ecuaciones y variables
    ecuaciones, variables = generar_ecuaciones(vectores)
    
    if determinanteActual != 0:
        print(f"\nEL CONJUNTO DE VECTORES ES UNA BASE porque el determinante es: {determinanteActual}")
    else:
        print(f"\nEL CONJUNTO DE VECTORES NO ES UNA BASE porque el determinante es 0")

    print("\nEcuaciones homogéneas planteadas:")
    for eq in ecuaciones:
        print(eq)
    
    print("\nSoluciones del sistema:")
    soluciones = solve(ecuaciones, variables, dict=True)  # Resolver el sistema de ecuaciones
    
    if soluciones:
        for idx, solucion in enumerate(soluciones, start=1):
            print(f"Solución {idx}: {solucion}")
    else:
        print("El sistema tiene infinitas soluciones o no tiene solución.")
    print("")
    return soluciones  # Retorna las soluciones para uso posterior


def rotar_vector(vector, angulo):
    theta = Rational(angulo) * pi / 180  # Convert angulo to radians
    rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])

    x, y = vector
    rotated_x = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y
    rotated_y = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y

    print(f"\nPrimer paso: Vectores resultante de aplicarle el angulo de {angulo}°:")
    print(rotation_matrix)
    # matriz de rotacion
    print(f"\nMATRIZ DE ROTACION{vector}:")
    print(f"[{rotation_matrix[0, 0]} {rotation_matrix[0, 1]}] * [{vector[0]}] = [{rotated_x}]")
    print(f"[{rotation_matrix[1, 0]} {rotation_matrix[1, 1]}]   [{vector[1]}]   [{rotated_y}]")
    
    return rotated_x, rotated_y

def mostrar_vectores_rotados(original_vectors, rotated_vectors):
    for i, (orig, rotated) in enumerate(zip(original_vectors, rotated_vectors)):
        frac_rotated = (rotated[0].simplify(), rotated[1].simplify())
        print(f"\nVector {chr(65 + i)}:")
        print(f"Original = {orig}")
        print(f"Rotado (exacto) = {rotated}")
        print(f"Rotado (fraccion si hace falta) = {frac_rotated}")
        print("\nCoordenadas en decimal:")
        print(f"x: {float(rotated[0])}")
        print(f"y: {float(rotated[1])}")

def graficar_vectores(original_vectors, rotated_vectors, angle):
    fig, ax = plt.subplots()

    # Plot original vectors
    for i, orig in enumerate(original_vectors):
        ax.plot(orig[0], orig[1], 'bo', label='Original' if i == 0 else "")  # blue point
        if i > 0:
            ax.plot([original_vectors[i-1][0], orig[0]], [original_vectors[i-1][1], orig[1]], 'b-')

    # Plot rotated vectors
    for i, rotated in enumerate(rotated_vectors):
        rotated_decimal = [float(rotated[0]), float(rotated[1])]
        ax.plot(rotated_decimal[0], rotated_decimal[1], 'ro', label='Rotado' if i == 0 else "")  # red point
        if i > 0:
            previous_rotated = [float(rotated_vectors[i-1][0]), float(rotated_vectors[i-1][1])]
            ax.plot([previous_rotated[0], rotated_decimal[0]], [previous_rotated[1], rotated_decimal[1]], 'r-')
    
    # Connect the first and last points to form closed figures if more than 2 vectors are provided
    if len(original_vectors) > 2:
        ax.plot([original_vectors[-1][0], original_vectors[0][0]], [original_vectors[-1][1], original_vectors[0][1]], 'b-')
        first_rotated = [float(rotated_vectors[0][0]), float(rotated_vectors[0][1])]
        last_rotated = [float(rotated_vectors[-1][0]), float(rotated_vectors[-1][1])]
        ax.plot([last_rotated[0], first_rotated[0]], [last_rotated[1], first_rotated[1]], 'r-')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.grid(True, which='both')
    ax.legend()  # Muestra las leyendas
    plt.show()    

def cambioBase():
    # Elegir el tipo de cambio de base
    print("\n1. Cambiar de Base B a Base C (Canónica)")
    print("2. Cambiar de Base C (Canónica) a Base B")
    opcion = input("Seleccione una opción: ")
    
    # Leer la base B
    print("\nIngrese la base B (2 vectores en R^2 o 3 vectores en R^3).")
    base_B = get_vectors()
    
    # Crear la matriz de la base B
    matriz_B = np.array(base_B).T  # Transponer para usar como columnas de la matriz
    
    # Validar que sea cuadrada
    if matriz_B.shape[0] != matriz_B.shape[1]:
        print(Fore.RED + "Error: La base debe ser cuadrada (dimensión coincidente).")
        return
    
    # Calcular las matrices de transición
    matriz_A_BC = matriz_B
    matriz_A_CB = np.linalg.inv(matriz_B)
    
    if opcion == "1":
        # Cambiar de Base B a Base C
        print("\nIngrese las coordenadas del vector en Base B (ej: 1,2):")
        vector_B = np.array(list(map(float, input().split(','))))
        
        # Transformar el vector
        vector_C = matriz_A_BC @ vector_B
        
        print("\n--- Resultados ---")
        print(f"Matriz de transición A_BC:\n{matriz_A_BC}")
        print(f"Vector en Base C (Canónica): {vector_C}")
    
    elif opcion == "2":
        # Cambiar de Base C a Base B
        print("\nIngrese las coordenadas del vector en Base C (Canónica) (ej: 1,2):")
        vector_C = np.array(list(map(float, input().split(','))))
        
        # Transformar el vector
        vector_B = matriz_A_CB @ vector_C
        
        print("\n--- Resultados ---")
        print(f"Matriz de transición A_CB:\n{matriz_A_CB}")
        print(f"Vector en Base B: {vector_B}")
    
    else:
        print(Fore.RED + "Opción inválida.")


# --------------- MAIN -------------------- 
menu = False

while menu == False:
    init(autoreset=True) 
    opcion = ""
    print(Fore.BLUE + "--- Calculadora algebra lineal ---")
    print("1. Encontrar base \n2. Cambio de base \n3. Transformaciones de Rotación \n4. Salir ")
    opcion = input("Digite la opción deseada: ")

    if(opcion == "1"):

        init(autoreset=True) 
        print(Fore.YELLOW + (" --- Encontrar base --- "))
        vectores = get_vectors()
        if len(vectores[0]) != len(vectores):
            print(Fore.RED + "Los vectores no forman una matriz cuadrada.")
        else:
            soluciones = calcularBase(vectores)
        
    elif(opcion == "2"):
        init(autoreset=True) 
        print(Fore.YELLOW + (" --- Cambio de base --- "))
        cambioBase()

    elif(opcion == "3"):
        init(autoreset=True) 
        print(Fore.YELLOW + (" --- Transformaciones de Rotación --- "))
        vectores = get_vectors()

        anguloInput = float(input("Ingresa el angulo de rotacion (En grados): "))
        print(f"\nRotando los vectores un angulo de {anguloInput}°:")
        vectoresRotados = [rotar_vector(v, anguloInput) for v in vectores]
        mostrar_vectores_rotados(vectores, vectoresRotados)
        graficar_vectores(vectores, vectoresRotados, anguloInput)

    elif(opcion == "4"):
        init(autoreset=True) 
        print(Fore.BLUE + (" Finalizando... "))
        menu = True

    else:
        print(Fore.YELLOW + "No se ha reconocido la opcion.")