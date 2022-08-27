#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
SLI Simulation 1
GWO
population size = 50
"""

import time
import resource
import multiprocessing

from niapy.task import OptimizationType, Task
from niapy.problems import Problem
from niapy.algorithms.basic import (
    GreyWolfOptimizer,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    HarrisHawksOptimization,
)

import numpy as np

# ************************************
population_size = 20
max_iters = 20
N_points = 300000 #1000000  # Numero de puntos aleatorios
# ************************************


def beta_AB(q_min, q_max):
    # Calculate the alpha and beta values for beta distribution
    # Cao2011,pg: 6
    # http://dx.doi.org/10.5772/45686
    q_range = q_max - q_min
    if q_range > 0 and q_range < pi / 2:
        a, b = 0.4, 0.4
    elif q_range >= pi / 2 and q_range < pi:
        u = (q_range / (5 * pi)) + 0.3
        a, b = u, u
    elif q_range >= pi and q_range < 3 * pi / 2:
        u = (3 * q_range / (5 * pi)) - 0.1
        a, b = u, u
    elif q_range >= (3 * pi / 2) and q_range <= 2 * pi:
        u = (2 * q_range / (5 * pi)) + 0.2
        a, b = u, u

    return (a, b)


def q_beta(q_min, q_max, N):
    if q_min == 0 and q_max == 0:
        q = np.zeros([N, 1])
    else:
        a, b = beta_AB(q_min, q_max)
        q = q_min + (q_max - q_min) * np.random.beta(a, b, N)
        q = np.reshape(q, [N, 1])
    return q


pi = 3.141592653589793
q1_min = -90 * (pi / 180)
q1_max = 90 * (pi / 180)

q2_min = -60 * (pi / 180)
q2_max = 120 * (pi / 180)

q3_min = -60 * (pi / 180)
q3_max = 120 * (pi / 180)

q4_min = -180 * (pi / 180)
q4_max = 180 * (pi / 180)

q5_min = -90 * (pi / 180)
q5_max = 90 * (pi / 180)

q6_min = -180 * (pi / 180)
q6_max = 180 * (pi / 180)

q_ranges = np.matrix(
    [
        [q1_min, q1_max],
        [q2_min, q2_max],
        [q3_min, q3_max],
        [q4_min, q4_max],
        [q5_min, q5_max],
        [q6_min, q6_max],
    ]
)

# G6R.q_beta(self, q_min, q_max, N)

n_lines = 50  # Slices x
n_slicez = 40  # Slices Z
hueco = 5  # n veces el ancho de la slice para determinar vacio
offset = 0  # Ofset para el area total

# Get q joints random beta distributted vector

# ***************************************************************************************
q6_beta = q_beta(q_ranges[5, 0], q_ranges[5, 1], N_points)
q5_beta = q_beta(q_ranges[4, 0], q_ranges[4, 1], N_points)
q4_beta = q_beta(q_ranges[3, 0], q_ranges[3, 1], N_points)
q3_beta = q_beta(q_ranges[2, 0], q_ranges[2, 1], N_points)
q2_beta = q_beta(q_ranges[1, 0], q_ranges[1, 1], N_points)
q1_beta = q_beta(q_ranges[0, 0], q_ranges[0, 1], N_points)

# ****************************************************************************************

# Calculate Sin an Cosine q joints values
# s6 = np.sin(q6_beta)
# c6 = np.cos(q6_beta)
s5 = np.sin(q5_beta)
c5 = np.cos(q5_beta)
s4 = np.sin(q4_beta)
c4 = np.cos(q4_beta)
s3 = np.sin(q3_beta)
c3 = np.cos(q3_beta)
s2 = np.sin(q2_beta)
c2 = np.cos(q2_beta)
s1 = np.sin(q1_beta)
c1 = np.cos(q1_beta)
c23 = np.cos(q2_beta + q3_beta)
s23 = np.sin(q2_beta + q3_beta)

q6_beta = q6_beta.flatten()
q5_beta = q5_beta.flatten()
q4_beta = q4_beta.flatten()
q3_beta = q3_beta.flatten()
q2_beta = q2_beta.flatten()
q1_beta = q1_beta.flatten()

# our custom Problem class
class G6RSLI(Problem):
    def __init__(self, dimension, lower=0.10, upper=1, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    # def SLI(self):

    #     px = self.d6 * s1 * s4 * s5 + c1 * (
    #         self.a2 * c2 + c23 * (self.d4 + self.d6 * c5) - self.d6 * c4 * s23 * s5
    #     )
    #     py = (
    #         c23 * (self.d4 + self.d6 * c5) * s1
    #         - self.d6 * (c3 * c4 * s1 * s2 + c1 * s4) * s5
    #         + c2 * s1 * (self.a2 - self.d6 * c4 * s3 * s5)
    #     )
    #     pz = (
    #         self.d1
    #         + self.a2 * s2
    #         + (self.d4 + self.d6 * c5) * s23
    #         + self.d6 * c23 * c4 * s5
    #     )

    #     px = px.flatten()  # aplanar array
    #     py = py.flatten()
    #     pz = pz.flatten()

    #     # Grid generator
    #     px_min = min(px)
    #     px_max = max(px)
    #     # py_min = min(py)
    #     # py_max = max(py)
    #     pz_min = min(pz)
    #     pz_max = max(pz)

    #     # Volume Box Calculation
    #     px_box = np.linspace(px_min - offset, px_max + offset, n_lines)
    #     # py_box = np.linspace(py_min - offset, py_max + offset, n_lines)
    #     pz_box = np.linspace(pz_min - offset, pz_max + offset, n_slicez)

    #     # Volume total
    #     # Area_Box = (
    #     #     ((py_max + offset) - (py_min - offset))
    #     #     * ((px_max + offset) - (px_min - offset))
    #     #     * ((pz_max + offset) - (pz_min - offset))
    #     # )

    #     # Matriz de posiciones
    #     Pos = np.matrix([px, py, pz])
    #     Pos = Pos.T

    #     # Tamaño de  X y Z slices
    #     z_slice = (pz_max - pz_min) / n_slicez  # Z slice
    #     xx_slice = (px_max - px_min) / n_lines  # X slice
    #     # print(xx_slice)

    #     # Auxilar list
    #     Area_slices = []

    #     # Para cada Z slices
    #     for j in range(n_slicez - 1):
    #         # Se inicializan las variables para guardar
    #         Area_h = 0  # Area de secciones vacias
    #         void_points_max = []  # Listas para guardar los puntos
    #         void_points_min = []
    #         fill_points_max = []
    #         fill_points_min = []

    #         # Se delimita el slice de Z
    #         slice_Zmin = pz_box[j]
    #         slice_Zmax = pz_box[j + 1]

    #         # Se genera el plano XY
    #         xy_plane = Pos[np.where(Pos[:, 2] >= slice_Zmin)[0]]
    #         xy_plane = xy_plane[np.where(xy_plane[:, 2] < slice_Zmax)[0]]
    #         # xz_plane = xy_plane
    #         xy_plane = np.array(xy_plane[:, 0:2])

    #         # Si el plano tiene puntos
    #         if xy_plane.size > 0:

    #             # Se calcula el area XY  en base a n_lines-1 secciones en X
    #             for i in range(n_lines - 1):
    #                 # limites de las secciones en x
    #                 box_xmin = px_box[i]
    #                 box_xmax = px_box[i] + xx_slice
    #                 # print("xmin xmax[", box_xmin, box_xmax, "]")

    #                 # Se filtran los puntos donde x >= box_xmin
    #                 x_slice = xy_plane[np.where(xy_plane[:, 0] >= box_xmin)[0]]

    #                 # Se filtran los valores anteriores donde x < box_xmax
    #                 x_slice = x_slice[np.where(x_slice[:, 0] < box_xmax)[0]]

    #                 # Si la slice en X tiene puntos
    #                 if x_slice.size > 0:

    #                     # Se busca el max y min en Y
    #                     y_max = float(max(x_slice[:, 1]))
    #                     y_min = float(min(x_slice[:, 1]))
    #                     # print(y_max, y_min)

    #                     # Se busca el puntos xy del max_y y min_y
    #                     point_ymax = np.array(
    #                         x_slice[np.where(x_slice[:, 1] == y_max)[0]]
    #                     )
    #                     point_ymin = np.array(
    #                         x_slice[np.where(x_slice[:, 1] == y_min)[0]]
    #                     )
    #                     # print("point y max", point_ymax)

    #                     # Se guardan los puntos XY en una lista
    #                     fill_points_max.append(point_ymax.flatten())
    #                     fill_points_min.append(point_ymin.flatten())

    #                     # Se ordenan los valores en Y de la seccion
    #                     R = np.array(x_slice)
    #                     R = R[R[:, 1].argsort()]

    #                     # Se calcula la distancia entre puntos para determinar si hay aberturas o huecos
    #                     aux_d = R[0, 1]  # 1ra posicion de Y mayor a menor

    #                     for j in R[:, 1]:  # Todas las puntos en el slice
    #                         largo = j - aux_d  # Largo de la seccion vacia
    #                         # print("Distancia: ", largo)

    #                         if largo >= (
    #                             hueco * xx_slice
    #                         ):  # Si el espacio vacio es mayor a nveces el ancho de la slice en x
    #                             # Se gurdan los puntos consecutivos del espacio vacio
    #                             point_yhmax = np.array(
    #                                 x_slice[np.where(x_slice[:, 1] == j)[0]]
    #                             )
    #                             point_yhmin = np.array(
    #                                 x_slice[np.where(x_slice[:, 1] == aux_d)[0]]
    #                             )
    #                             # Se calcula el area del vacio b*h
    #                             area_h = largo * xx_slice

    #                             Area_h = (
    #                                 Area_h + area_h
    #                             )  # Se guarda el Arae vacia de cada slice

    #                             # Se guardan los puntos del area vacia
    #                             void_points_max.append(point_yhmax.flatten())
    #                             void_points_min.append(point_yhmin.flatten())
    #                         aux_d = j  # Dato para siguiente iteracion

    #             # Puntos Bordes exteriores e interiores

    #             # *************************************************************
    #             void_points_max = np.matrix(void_points_max)
    #             void_points_min = np.matrix(void_points_min)
    #             fill_points_max = np.matrix(fill_points_max)
    #             fill_points_min = np.matrix(fill_points_min)
    #             # *************************************************************

    #             # Calculo de area
    #             # Vector con los puntos minimos y maximos unidos y ordenados
    #             borde = np.vstack(
    #                 [fill_points_max, fill_points_min[::-1], fill_points_max[0, :]]
    #             )

    #             # Acomodo de putnos  (xi*yi+1 - xi+1*yi)
    #             xi_plus1 = borde[1::, 0]
    #             yi = borde[:-1, 1]
    #             yi_plus1 = borde[1::, 1]
    #             xi = borde[:-1, 0]

    #             aux_a, Area = 0, 0  # Variables auxiliares

    #             # Area = 0.5 * SUMATORIA (xi*yi+1 - xi+1*yi)
    #             for i in range(xi.size):
    #                 aux_a = xi[i] * yi_plus1[i] - xi_plus1[i] * yi[i]
    #                 Area = Area + aux_a

    #             Area = float(0.5 * (abs(Area)) - 1 * Area_h)  # Area llena - area vacia

    #             # Se concatenan las j areas calculadas
    #             Area_slices.append(Area)

    #     # Calculo del Volumen
    #     # Trapezoidal integration method
    #     Volumen = z_slice * (
    #         sum(Area_slices) - 0.5 * (Area_slices[0] + Area_slices[-1])
    #     )

    #     L = self.d4 + self.d6 + self.a2 + self.d1
    #     SLI = L / (Volumen) ** (1 / 3)
    #     return SLI

    def _evaluate(self, x):

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        g1 = -((x1 / x2) - 1.1)
        g2 = (x1 / x2) - 2.0
        g3 = -((x2 / x3) - 1.1)
        g4 = (x2 / x3) - 2.0
        pp = 10 ^ 7
        g = [g1, g2, g3, g4]
        penalty = [0, 0, 0, 0]
        for i in range(4):
            if g[i] > 0:
                penalty[i] = pp * g[i]

        d1 = 30
        a2 = x1 * 100
        d4 = x2 * 100
        d6 = x3 * 100
        # Links = [d1, a2, 0, d4, 0, d6]

        # SLIx = self.SLI + sum(penalty)
        # return SLIx

        # a2 = (float(x1 * 50 + 30)) / 1
        # d4 = (float(x2 * 30 + 20)) / 1
        # d6 = (float(x3 * 30 + 10)) / 1
        # a2 = (int([0.146907* 70 + 20)) / 10
        # d4 = (int(0.727662* 50 + 10)) / 10
        # d6 = (int(0.71836189 * 30 + 10)) / 1
        # X Y Z position
        # px = d6 * s1 * s4 * s5 + c1 * (            a2 * c2 + c23 * (d4 + d6 * c5) - d6 * c4 * s23 * s5        )
        #py = (            c23 * (d4 + d6 * c5) * s1            - d6 * (c3 * c4 * s1 * s2 + c1 * s4) * s5            + c2 * s1 * (a2 - d6 * c4 * s3 * s5))
        #pz = d1 + a2 * s2 + (d4 + d6 * c5) * s23 + d6 * c23 * c4 * s5
        px = a2*c1*c2 + d4*c1*s23 + d6*(c1*(c23*c4*s5 + s23*c5) + s1*s4*s5)
        py = a2*s1*c2 + d4*s1*s23 + d6*(s1*(c23*c4*s5 + s23*c5) - c1*s4*s5)
        pz = a2*s2 - d4*c23 + d6*(s23*c4*s5 - c23*c5) + d1

        px = px.flatten()  # aplanar array
        py = py.flatten()
        pz = pz.flatten()

        # Grid generator
        px_min = min(px)
        px_max = max(px)
        # py_min = min(py)
        # py_max = max(py)
        pz_min = min(pz)
        pz_max = max(pz)

        # Volume Box Calculation
        px_box = np.linspace(px_min - offset, px_max + offset, n_lines)
        # py_box = np.linspace(py_min - offset, py_max + offset, n_lines)
        pz_box = np.linspace(pz_min - offset, pz_max + offset, n_slicez)

        # Volume total
        # Area_Box = (
        #     ((py_max + offset) - (py_min - offset))
        #     * ((px_max + offset) - (px_min - offset))
        #     * ((pz_max + offset) - (pz_min - offset))
        # )

        # Matriz de posiciones
        Pos = np.matrix([px, py, pz])
        Pos = Pos.T

        # Tamaño de  X y Z slices
        z_slice = (pz_max - pz_min) / n_slicez  # Z slice
        xx_slice = (px_max - px_min) / n_lines  # X slice
        # print(xx_slice)

        # Auxilar list
        Area_slices = []

        # Para cada Z slices
        for j in range(n_slicez - 1):
            # Se inicializan las variables para guardar
            Area_h = 0  # Area de secciones vacias
            void_points_max = []  # Listas para guardar los puntos
            void_points_min = []
            fill_points_max = []
            fill_points_min = []

            # Se delimita el slice de Z
            slice_Zmin = pz_box[j]
            slice_Zmax = pz_box[j + 1]

            # Se genera el plano XY
            xy_plane = Pos[np.where(Pos[:, 2] >= slice_Zmin)[0]]
            xy_plane = xy_plane[np.where(xy_plane[:, 2] < slice_Zmax)[0]]
            # xz_plane = xy_plane
            xy_plane = np.array(xy_plane[:, 0:2])

            # Si el plano tiene puntos
            if xy_plane.size > 0:

                # Se calcula el area XY  en base a n_lines-1 secciones en X
                for i in range(n_lines - 1):
                    # limites de las secciones en x
                    box_xmin = px_box[i]
                    box_xmax = px_box[i] + xx_slice
                    # print("xmin xmax[", box_xmin, box_xmax, "]")

                    # Se filtran los puntos donde x >= box_xmin
                    x_slice = xy_plane[np.where(xy_plane[:, 0] >= box_xmin)[0]]

                    # Se filtran los valores anteriores donde x < box_xmax
                    x_slice = x_slice[np.where(x_slice[:, 0] < box_xmax)[0]]

                    # Si la slice en X tiene puntos
                    if x_slice.size > 0:

                        # Se busca el max y min en Y
                        y_max = float(max(x_slice[:, 1]))
                        y_min = float(min(x_slice[:, 1]))
                        # print(y_max, y_min)

                        # Se busca el puntos xy del max_y y min_y
                        point_ymax = np.array(
                            x_slice[np.where(x_slice[:, 1] == y_max)[0]]
                        )
                        point_ymin = np.array(
                            x_slice[np.where(x_slice[:, 1] == y_min)[0]]
                        )
                        # print("point y max", point_ymax)

                        # Se guardan los puntos XY en una lista
                        fill_points_max.append(point_ymax.flatten())
                        fill_points_min.append(point_ymin.flatten())

                        # Se ordenan los valores en Y de la seccion
                        R = np.array(x_slice)
                        R = R[R[:, 1].argsort()]

                        # Se calcula la distancia entre puntos para determinar si hay aberturas o huecos
                        aux_d = R[0, 1]  # 1ra posicion de Y mayor a menor

                        for j in R[:, 1]:  # Todas las puntos en el slice
                            largo = j - aux_d  # Largo de la seccion vacia
                            # print("Distancia: ", largo)

                            if largo >= (
                                hueco * xx_slice
                            ):  # Si el espacio vacio es mayor a nveces el ancho de la slice en x
                                # Se gurdan los puntos consecutivos del espacio vacio
                                point_yhmax = np.array(
                                    x_slice[np.where(x_slice[:, 1] == j)[0]]
                                )
                                point_yhmin = np.array(
                                    x_slice[np.where(x_slice[:, 1] == aux_d)[0]]
                                )
                                # Se calcula el area del vacio b*h
                                area_h = largo * xx_slice

                                Area_h = (
                                    Area_h + area_h
                                )  # Se guarda el Arae vacia de cada slice

                                # Se guardan los puntos del area vacia
                                void_points_max.append(point_yhmax.flatten())
                                void_points_min.append(point_yhmin.flatten())
                            aux_d = j  # Dato para siguiente iteracion

                # Puntos Bordes exteriores e interiores

                # *************************************************************
                void_points_max = np.matrix(void_points_max)
                void_points_min = np.matrix(void_points_min)
                fill_points_max = np.matrix(fill_points_max)
                fill_points_min = np.matrix(fill_points_min)
                # *************************************************************

                # Calculo de area
                # Vector con los puntos minimos y maximos unidos y ordenados
                borde = np.vstack(
                    [fill_points_max, fill_points_min[::-1], fill_points_max[0, :]]
                )

                # Acomodo de putnos  (xi*yi+1 - xi+1*yi)
                xi_plus1 = borde[1::, 0]
                yi = borde[:-1, 1]
                yi_plus1 = borde[1::, 1]
                xi = borde[:-1, 0]

                aux_a, Area = 0, 0  # Variables auxiliares

                # Area = 0.5 * SUMATORIA (xi*yi+1 - xi+1*yi)
                for i in range(xi.size):
                    aux_a = xi[i] * yi_plus1[i] - xi_plus1[i] * yi[i]
                    Area = Area + aux_a

                Area = float(0.5 * (abs(Area)) - 1 * Area_h)  # Area llena - area vacia

                # Se concatenan las j areas calculadas
                Area_slices.append(Area)

        # Calculo del Volumen
        # Trapezoidal integration method
        Volumen = z_slice * (
            sum(Area_slices) - 0.5 * (Area_slices[0] + Area_slices[-1])
        )

        L = d4 + d6 + a2 + d1
        SLI = (L / (Volumen) ** (1 / 3)) + sum(penalty)
        return SLI
        # print("SLI: ", SLI)
        # print("a2 - d4 - d6 : ", a2, d4, d6)

        # scores.append([SLI, Volumen, a2, d4, d6])


G6R_sli_opt = G6RSLI(dimension=3)

# GWO a=expo


def GWO_SLI_Optimiazation():
    print("GWO start\n")
    task = Task(
        problem=G6R_sli_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MINIMIZATION,
    )
    algo_gwo = GreyWolfOptimizer(population_size)
    best_gwo = algo_gwo.run(task)
    best2 = [str(best_gwo)]

    print("Best solution  GWO", best_gwo)
    np.savetxt("GWO_best.txt", best2, fmt="%s")
    np.savetxt(
        "GWO_convergence_data_values.cvs", task.convergence_data(), delimiter=","
    )
    print("GWO done\n")


def GA_SLI_Optimization():

    print("GA start\n")
    task = Task(
        problem=G6R_sli_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MINIMIZATION,
    )
    algo_ga = GeneticAlgorithm(population_size)
    best_ga = algo_ga.run(task)
    best2 = [str(best_ga)]
    print("Best solution  GA", best_ga)
    np.savetxt("GA_best.txt", best2, fmt="%s")
    np.savetxt("GA_convergence_data_values.cvs", task.convergence_data(), delimiter=",")
    print("GA done\n")


def PSO_SLI_Optimization():
    print("PSO start\n")
    task = Task(
        problem=G6R_sli_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MINIMIZATION,
    )
    algo_pso = ParticleSwarmOptimization(population_size)
    best_pso = algo_pso.run(task)
    best2 = [str(best_pso)]
    print("Best solution  PSO", best_pso)
    np.savetxt("PSO_best.txt", best2, fmt="%s")
    np.savetxt(
        "PSO_convergence_data_values.cvs", task.convergence_data(), delimiter=","
    )
    print("PSO done\n")


def HHO_SLI_Optimization():
    print("HHO start\n")
    task = Task(
        problem=G6R_sli_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MINIMIZATION,
    )
    algo_hho = HarrisHawksOptimization(population_size)
    best_hho = algo_hho.run(task)
    best2 = [str(best_hho)]
    print("Best solution  HHO", best_hho)
    np.savetxt("HHO_best.txt", best2, fmt="%s")
    np.savetxt(
        "HHO_convergence_data_values.cvs", task.convergence_data(), delimiter=","
    )
    print("HHO done\n")


def Algorithm_selector(algo_type):
    if algo_type == 0:
        time_start = time.perf_counter()  # Timer Start
        GWO_SLI_Optimiazation()
        time_elapsed = time.perf_counter() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("GWO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
        time_memory = [time_elapsed, memMb]
        np.savetxt("GWO_time.txt", time_memory, fmt="%s")
    elif algo_type == 1:
        time_start = time.perf_counter()  # Timer Start
        GA_SLI_Optimization()
        time_elapsed = time.perf_counter() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("GA %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
        time_memory = [time_elapsed, memMb]
        np.savetxt("GA_time.txt", time_memory, fmt="%s")
    elif algo_type == 2:
        time_start = time.perf_counter()  # Timer Start
        PSO_SLI_Optimization()
        time_elapsed = time.perf_counter() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("PSO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
        time_memory = [time_elapsed, memMb]
        np.savetxt("PSO_time.txt", time_memory, fmt="%s")
    elif algo_type == 3:
        time_start = time.perf_counter()  # Timer Start
        HHO_SLI_Optimization()
        time_elapsed = time.perf_counter() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("HHO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
        time_memory = [time_elapsed, memMb]
        np.savetxt("HHO_time.txt", time_memory, fmt="%s")


time_start = time.perf_counter()  # Timer Start
## Multiprocessing
pool = multiprocessing.Pool(4)
pool.map(Algorithm_selector, range(4))
pool.close()


time_elapsed = time.perf_counter() - time_start
memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
print("COMPLETE PROG %5.5f secs %5.5f MByte" % (time_elapsed, memMb))
time_memory = [time_elapsed, memMb]
np.savetxt("TOTAL_time.txt", time_memory, fmt="%s")
