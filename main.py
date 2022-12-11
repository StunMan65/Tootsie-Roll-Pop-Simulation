import os

from scipy.fft import fft, ifft
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image


# Measurements:
#
# Ring Diameter: 25.967 mm
# Inner Diameter: 23.733 mm
# Total Height: 33.65 mm
# Ring Height: 14.367 mm
# Shape: Top of Sphere - Cylinder - Bottom of Sphere
# Chord Height: (T - h) / 2 = 9.6415 mm
# Sphere Radius: L^2 / 8h + h / 2 = 12.123 mm

def circle(radius, num_points, x_off=0.0, y_off=0.0):
    return [(radius * np.cos((2.0 * np.pi * x) / num_points) + x_off, radius * np.sin((2.0 * np.pi * x) / num_points) + y_off) for x in range(num_points)]


def rectangle(width, height, num_points, x_off=0.0, y_off=0.0):
    x_points = int(np.ceil(width / (width + height) * num_points * 0.5))
    y_points = int(np.ceil(height / (width + height) * num_points * 0.5))
    right = [(x_off + 0.5 * width, y + y_off) for y in np.linspace(-0.5 * height, 0.5 * height, y_points)]
    left = [(x_off + -0.5 * width, y + y_off) for y in np.linspace(-0.5 * height, 0.5 * height, y_points)]
    top = [(x_off + x, 0.5 * height + y_off) for x in np.linspace(-0.5 * width, 0.5 * width, x_points)]
    bottom = [(x_off + x, -0.5 * height + y_off) for x in np.linspace(-0.5 * width, 0.5 * width, x_points)]
    return right + top + left + bottom


def find_volume_rotated(points, dh):
    points = [x for x in points if x[0] > 0.0]

    def radius(y):
        ys = np.array([y for x, y in points])
        xs = np.array([x for x, y in points])
        index = np.argmin(np.abs(ys - y))
        right_index = index + 1
        if right_index >= len(points):
            return xs[index]
        left_index = index - 1
        if left_index < 0:
            return xs[index]
        if abs(ys[right_index] - y) < abs(ys[left_index] - y):
            first_index = index
            second_index = right_index
        else:
            first_index = left_index
            second_index = index
        if abs(ys[first_index] - ys[second_index]) > 1e-8:
            rad = (y - ys[second_index]) / (ys[first_index] - ys[second_index]) * (xs[first_index] - xs[second_index]) + xs[second_index]
            return rad
        return xs[first_index]

    target_height = np.amin([y for x, y in points])
    y_max = np.amax([y for x, y in points])
    volume = 0.0
    while target_height < y_max:
        r = radius(target_height)
        volume += r * r * dh
        target_height += dh
    volume *= np.pi
    return volume

def find_surface_area_rotated(points, dh):
    points = [x for x in points if x[0] > 0.0]

    def radius(y):
        ys = np.array([y for x, y in points])
        xs = np.array([x for x, y in points])
        index = np.argmin(np.abs(ys - y))
        right_index = index + 1
        if right_index >= len(points):
            return xs[index]
        left_index = index - 1
        if left_index < 0:
            return xs[index]
        if abs(ys[right_index] - y) < abs(ys[left_index] - y):
            first_index = index
            second_index = right_index
        else:
            first_index = left_index
            second_index = index
        if abs(ys[first_index] - ys[second_index]) > 1e-8:
            rad = (y - ys[second_index]) / (ys[first_index] - ys[second_index]) * (xs[first_index] - xs[second_index]) + xs[second_index]
            return rad
        return xs[first_index]

    target_height = np.amin([y for x, y in points])
    y_max = np.amax([y for x, y in points])
    surface_area = 0.0
    r1 = radius(target_height)
    while target_height+dh < y_max:
        r2 = r1
        r1 = radius(target_height+dh)
        if r2 > r1:
            y = (r1*dh)/(r2-r1)
            surface_area += (r2*np.sqrt(np.square(y+dh)+r2*r2) - r1*np.sqrt(y*y+r1*r1))
        elif r2 == r1:
            surface_area += 2*r1*dh
        else:
            y = (r2 * dh) / (r1 - r2)
            surface_area += (r1 * np.sqrt(np.square(y + dh) + r1 * r1) - r2 * np.sqrt(y * y + r2 * r2))
        target_height += dh
    surface_area *= np.pi
    return surface_area

def generate_shape(sample_points):
    top_circle = lambda num_points: circle(12.123, num_points, 0.0, 4.702)
    bottom_circle = lambda num_points: circle(12.123, num_points, 0.0, -4.702)
    middle = lambda num_points: rectangle(26.275, 14.367, num_points)
    start_shape = [x for x in top_circle(sample_points) if x[1] >= 0.5 * 14.367] + [x for x in bottom_circle(sample_points) if x[1] <= -0.5 * 14.367] + [x for x in
                                                                                                                                                         middle(sample_points) if
                                                                                                                                                         abs(x[0]) >= 23.733 * 0.5]
    angles = [np.arctan(np.divide(y, x)) if x > 0 else np.arctan(np.divide(y, x)) + np.pi for x, y in start_shape]
    angles = np.array([x if x > 0 else x + 2 * np.pi for x in angles])
    end_shape = []
    for i in range(sample_points):
        target_angle = (2.0 * np.pi * i) / sample_points
        index = np.argmin(np.abs(angles - target_angle))
        right_index = index + 1
        if right_index >= len(angles):
            right_index = 0
        left_index = index - 1
        if abs(angles[right_index] - target_angle) < abs(angles[left_index] - target_angle):
            first_point = index
            second_point = right_index
        else:
            first_point = left_index
            second_point = index
        tantheta = np.tan(target_angle)
        x1, y1 = start_shape[first_point]
        x2, y2 = start_shape[second_point]
        coeff_1 = (y2 - x2 * tantheta) / (x1 * tantheta - y1 - x2 * tantheta + y2)
        coeff_2 = 1 - coeff_1
        end_shape.append(complex(coeff_1 * x1 + coeff_2 * x2, coeff_1 * y1 + coeff_2 * y2))
    return end_shape

def loss(pred, true):
    return np.sum(np.square(np.array(pred) - np.array(true))) / len(pred)

# Rate of Dissolution: dm/dt = A * (D / d) * (Cs - Cb)
# A = surface area of solute particle
# D = diffusion coefficient
# d = thickness of concentration gradient
# Cs = particle surface concentration
# Cb = concentration in the bulk solution

SAMPLE_RATE = 500
TOTAL_TIME = 390  # in secs
TIME_PER_CHECK = 30  # in secs
TIME_GRANULARITY = 1  # in secs
SOLUTE_SOLUBILITY = 0.0023  # @37C, sucrose = 230g / 100g; 230g / 100 cm^3; 230g / 100000 mm^3; 0.0023 g/mm^3
SOLUTE_DENSITY = 0.00159  # 1.59 g/cm^3; 0.00159 g/mm^3
SOLVENT_RATE = 28.33  # 1.5–2.0 mL/min; 1700 mm^3/min; 28.33 mm^3/s
BEGINNING_SOLVENT = 1190  # 1.19 mL; 1190 mm^3 https://pubmed.ncbi.nlm.nih.gov/6584462/

NUM_CHECKS = int(TOTAL_TIME // TIME_PER_CHECK)
NUM_ITER = int(TIME_PER_CHECK // TIME_GRANULARITY)

def run_sim(alpha, diff_coeff, trp, log = False, graph=False):
    xs = [x.real for x in trp]
    ys = [x.imag for x in trp]
    predicted = [[0.0, 0.0] for _ in range(NUM_CHECKS)]
    end = (NUM_CHECKS - 1, NUM_ITER - 1)
    cumulative_scaling = 1.0
    volume_change = 0.0
    volume_before = 0.0
    if graph:
        data = []
    for k in range(NUM_CHECKS):
        dissolved_solid = 0.0
        solute_volume = BEGINNING_SOLVENT
        for j in range(NUM_ITER):
            points = list(zip(xs, ys))
            try:
                volume = find_volume_rotated(points, dh=0.05)
                surface_area = find_surface_area_rotated(points, dh=0.05)
            except ValueError:
                end = (k, j)
                break
            solute_volume += SOLVENT_RATE * TIME_GRANULARITY
            dissolved_solid += SOLUTE_DENSITY * volume_change
            volume_change = surface_area * diff_coeff * (SOLUTE_SOLUBILITY - dissolved_solid/solute_volume)
            pop_scaling = np.power((volume - volume_change) / volume, 1.0 / 3.0)
            cumulative_scaling *= pop_scaling
            if log:
                plt.clf()
                plt.cla()
            new_alpha = alpha * (k*NUM_ITER + j)
            scaling = [np.power(np.e, new_alpha * i * i) for i in range(len(tootsie_pop_fft)//2+1)]
            scaling = np.array([scaling[0]] + scaling[1:] + scaling[1:])[:-1]
            new_tootsie_pop_fft = np.multiply(scaling, tootsie_pop_fft)
            new_tootsie_pop = ifft(new_tootsie_pop_fft)
            xs = [x.real for x in new_tootsie_pop]
            ys = [x.imag for x in new_tootsie_pop]
            if volume_before == 0.0:
                volume_scale = 1.0
            else:
                volume_scale = volume / volume_before
            volume_before = volume
            xs = np.array(xs) * volume_scale * cumulative_scaling
            ys = np.array(ys) * volume_scale * cumulative_scaling #(base_height / current_height) * cumulative_scaling
            if log:
                plt.plot(xs, ys, 'o')
                plt.axis('scaled')
                plt.xlim(-15, 15)
                plt.ylim(-20, 20)
                plt.savefig(f'shapes/{k*NUM_ITER + j}.png')
            x_diameter = np.max(xs) - np.min(xs)
            y_diameter = np.max(ys) - np.min(ys)
            if graph:
                data.append([volume, surface_area, volume_change, solute_volume, dissolved_solid, x_diameter, y_diameter])
        if end != (NUM_CHECKS-1, NUM_ITER-1):
            break
        predicted[k] = [y_diameter, x_diameter]
    if graph:
        return predicted, data
    if not log:
        return predicted
    return predicted, end

LOG = True
GRAPH = True
# To optimize:
current_alpha = -0.00001005
current_diff_coeff = 3.44468652  # equal to (D / d)
tootsie_pop = generate_shape(SAMPLE_RATE)
tootsie_pop_fft = np.array(fft(tootsie_pop))

ACCEPTABLE_ERROR_CHANGE = 1e-8
MOVE_ALPHA_BIAS = -2e-06
MOVE_DIFF_BIAS = 0.8
DX = 1e-12
LEARNING_RATE = 0.5
FIT_DATA = [[34.2, 25.8],
            [33.6, 25.7],
            [33.3, 25.3],
            [33.0, 24.6],
            [32.6, 24.2],
            [31.4, 23.6],
            [30.9, 23.3],
            [30.8, 22.5],
            [30.2, 21.8],
            [29.6, 21.2],
            [28.6, 20.7],
            [28.0, 20.2],
            [27.8, 19.3]]

old_alpha_loss = 500000
new_alpha_loss = 100000
old_coeff_loss = 500000
new_coeff_loss = 100000

#for i in range(-70, -50):
#    i /= 1e5
#    loss_am = loss(run_sim(i, current_diff_coeff, tootsie_pop), FIT_DATA)
#    print(f'{i}:{loss_am}')

#breakpoint()

while old_alpha_loss / new_alpha_loss > 1+ACCEPTABLE_ERROR_CHANGE or old_coeff_loss / new_coeff_loss > 1+ACCEPTABLE_ERROR_CHANGE:
    now = time.perf_counter()
    today = datetime.now()
    search_boundaries = [current_alpha / 3, 2 * current_alpha / 3, 4 * current_alpha / 3, 5 * current_alpha / 3]
    search_loss = [loss(run_sim(bound, current_diff_coeff, tootsie_pop), FIT_DATA) for bound in search_boundaries]
    print(f'# FINISHED BOUNDARIES RECALCULATION! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
    direction = 0
    bias_mult = 1.0
    for _ in range(15):
        now = time.perf_counter()
        today = datetime.now()
        old_alpha_loss = new_alpha_loss
        new_alpha_loss = np.min(search_loss)
        index_min = np.argmin(search_loss)
        print(f'# CURRENT BEST ALPHA: {search_boundaries[index_min]}')
        if index_min == 0:
            if direction == -1:
                bias_mult *= 0.5
            direction = 1
            search_boundaries = [min(bound - bias_mult * MOVE_ALPHA_BIAS, 0.0) for bound in search_boundaries]
            search_loss = [loss(run_sim(bound, current_diff_coeff, tootsie_pop), FIT_DATA) for bound in search_boundaries]
            print(
                f'# WAS OVERESTIMATED, MOVED BOUNDS DOWN! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
            continue
        elif index_min == 3:
            if direction == 1:
                bias_mult *= 0.5
            direction = -1
            search_boundaries = [bound + bias_mult * MOVE_ALPHA_BIAS for bound in search_boundaries]
            search_loss = [loss(run_sim(bound, current_diff_coeff, tootsie_pop), FIT_DATA) for bound in search_boundaries]
            print(
                f'# WAS UNDERESTIMATED, MOVED BOUNDS UP! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
            continue
        search_boundaries[0] = search_boundaries[index_min - 1]
        search_loss[0] = search_loss[index_min - 1]
        search_boundaries[3] = search_boundaries[index_min + 1]
        search_loss[3] = search_loss[index_min + 1]
        space = (search_boundaries[3] - search_boundaries[0]) / 3.0
        search_boundaries[1] = search_boundaries[0] + space
        search_boundaries[2] = search_boundaries[3] - space
        search_loss[1] = loss(run_sim(search_boundaries[1], current_diff_coeff, tootsie_pop), FIT_DATA)
        print(f'# FINISHED BOUNDARY 1 ALPHA SIM! LOSS: {search_loss[1]:5f}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
        now = time.perf_counter()
        today = datetime.now()
        search_loss[2] = loss(run_sim(search_boundaries[2], current_diff_coeff, tootsie_pop), FIT_DATA)
        print(f'# FINISHED BOUNDARY 2 ALPHA SIM! LOSS: {search_loss[2]:5f}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
    index_min = np.argmin(search_loss)
    current_alpha = search_boundaries[index_min]
    now = time.perf_counter()
    today = datetime.now()
    search_boundaries = [2 * current_diff_coeff / 3, 5 * current_diff_coeff / 6, 7 * current_diff_coeff / 6, 4 * current_diff_coeff / 3]
    search_loss = [loss(run_sim(current_alpha, bound, tootsie_pop), FIT_DATA) for bound in search_boundaries]
    print(
        f'# FINISHED BOUNDARIES RECALCULATION! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
    direction = 0
    bias_mult = 1.0
    for _ in range(15):
        now = time.perf_counter()
        today = datetime.now()
        old_coeff_loss = new_coeff_loss
        new_coeff_loss = np.min(search_loss)
        index_min = np.argmin(search_loss)
        print(f'# CURRENT BEST DIFF. COEFF: {search_boundaries[index_min]}')
        if index_min == 0:
            if direction == -1:
                bias_mult *= 0.5
            direction = 1
            search_boundaries = [bound - bias_mult * MOVE_DIFF_BIAS for bound in search_boundaries]
            search_loss = [loss(run_sim(current_alpha, bound, tootsie_pop), FIT_DATA) for bound in search_boundaries]
            print(
                f'# WAS OVERESTIMATED, MOVED BOUNDS DOWN! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
            continue
        elif index_min == 3:
            if direction == 1:
                bias_mult *= 0.5
            direction = -1
            search_boundaries = [bound + bias_mult * MOVE_DIFF_BIAS for bound in search_boundaries]
            search_loss = [loss(run_sim(current_alpha, bound, tootsie_pop), FIT_DATA) for bound in search_boundaries]
            print(
                f'# WAS UNDERESTIMATED, MOVED BOUNDS UP! LOSSES: {" ".join([str(round(x, 5)).ljust(5, "0") for x in search_loss])}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
            continue
        search_boundaries[0] = search_boundaries[index_min - 1]
        search_loss[0] = search_loss[index_min - 1]
        search_boundaries[3] = search_boundaries[index_min + 1]
        search_loss[3] = search_loss[index_min + 1]
        space = (search_boundaries[3] - search_boundaries[0]) / 3.0
        search_boundaries[1] = search_boundaries[0] + space
        search_boundaries[2] = search_boundaries[3] - space
        search_loss[1] = loss(run_sim(current_alpha, search_boundaries[1], tootsie_pop), FIT_DATA)
        print(f'# FINISHED BOUNDARY 1 DIFF. COEFF SIM! LOSS: {search_loss[1]:5f}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
        now = time.perf_counter()
        today = datetime.now()
        search_loss[2] = loss(run_sim(current_alpha, search_boundaries[2], tootsie_pop), FIT_DATA)
        print(f'# FINISHED BOUNDARY 2 DIFF. COEFF SIM! LOSS: {search_loss[2]:5f}; TIME ELAPSED: {time.perf_counter() - now:.2f}; STARTED: {str(today)}')
    index_min = np.argmin(search_loss)
    current_diff_coeff = search_boundaries[index_min]
    print(f'α: {current_alpha:.8f}; D: {current_diff_coeff:.8f}; LOSS: {search_loss[index_min]:.4f}')

print('# FINISHED WITH OPTIMIZATION!')

try:
    os.mkdir('shapes')
except:
    pass

if LOG:
    pred, end = run_sim(current_alpha, current_diff_coeff, tootsie_pop, log=True)
    images = []
    first_im = Image.open('shapes/0.png')
    for j in range(1, int(end[0]*NUM_ITER + end[1])):
       images.append(Image.open(f'shapes/{j}.png'))
    first_im.save('tootsie_pop2.gif', save_all=True, append_images=images, loop=0)

if GRAPH:
    titles = ['Volume (mm^3) vs. Time (s)', 'Surface Area (mm^2) vs. Time (s)', 'dVdt (mm^3 s^-1) vs. Time (s)', 'Amount of Solvent (mm^3) vs. Time (s)', 'Amount of Solute (g) vs. Time (s)', 'Diameter (mm) vs. Time (s)', 'Height (mm) vs. Time (s)']
    pred, data = run_sim(current_alpha, current_diff_coeff, tootsie_pop, graph=True)
    dydt = [(data[x+1][-1] - data[x-1][-1]) / 2 for x in range(1, len(data)-1)]
    dydt.insert(0, data[1][-1] - data[0][-1])
    dydt.append(data[-1][-1] - data[-2][-1])
    xs = list(range(len(data)))
    for index, title in enumerate(titles):
        plt.cla()
        plt.clf()
        plt.plot(xs, [x[index] for x in data])
        plt.title(title)
        plt.savefig(f'{title}.png')
    plt.cla()
    plt.clf()
    plt.plot(xs[10:], dydt[10:])
    plt.title('dhdt (mm s^-1) vs. Time (s)')
    plt.savefig(f'dhdt (mm s^-1) vs. Time (s).png')
