import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- USER CONFIGURATION ---
IR_STICK_FILE = 'opt.thymol.out.ir.stk'
RAMAN_STICK_FILE = 'opt.thymol.out.raman.stk'

# Plotting parameters
X_MIN = 0           # cm-1
X_MAX = 4000        # cm-1
NUM_POINTS = 16000  # 4 points per cm-1 over the 4000 cm-1 range
FWHM = 20.0         # Full-width at half-maximum for peak broadening in cm-1
FREQ_SCALING_FACTOR = 0.975
# -------------------------

def generate_spectrum(stk_file, x_axis, fwhm, scaling_factor):
    """
    Generates a convoluted spectrum from a .stk file.
    """
    y_axis = np.zeros_like(x_axis)
    try:
        sticks = np.loadtxt(stk_file)
    except IOError:
        print(f"Error: Could not find or read the file {stk_file}")
        return y_axis
    frequencies = sticks[:, 0] * scaling_factor
    intensities = sticks[:, 1]
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for freq, intensity in zip(frequencies, intensities):
        y_axis += intensity * np.exp(-((x_axis - freq)**2) / (2 * sigma**2))
    return y_axis

# --- Main Script ---
# 1. Create the x-axis (wavenumber range)
x_axis = np.linspace(X_MIN, X_MAX, NUM_POINTS)

# 2. Generate the raw IR and Raman spectra
print(f"-> Generating IR spectrum from {IR_STICK_FILE}...")
ir_y_raw = generate_spectrum(IR_STICK_FILE, x_axis, FWHM, FREQ_SCALING_FACTOR)
print(f"-> Generating Raman spectrum from {RAMAN_STICK_FILE}...")
raman_y_raw = generate_spectrum(RAMAN_STICK_FILE, x_axis, FWHM, FREQ_SCALING_FACTOR)

# 3. Normalize data and apply vertical offset to IR
ir_max = np.max(ir_y_raw) if np.max(ir_y_raw) > 1e-6 else 1.0
raman_max = np.max(raman_y_raw) if np.max(raman_y_raw) > 1e-6 else 1.0

# Normalize Raman data to the range [0, 1] for the bottom of the plot
raman_y_normalized = raman_y_raw / raman_max

# Convert IR to transmittance [0, 1] and then shift it up to range [1.2, 2.2]
ir_y_transmittance = 1.0 - (ir_y_raw / ir_max)
ir_y_offset = ir_y_transmittance + 1.2

# 4. Plotting
fig, ax1 = plt.subplots(figsize=(16, 8))
ax2 = ax1.twinx()

# --- Set IDENTICAL y-limits for both axes ---
Y_MIN, Y_MAX = -0.1, 2.3
ax1.set_ylim(Y_MIN, Y_MAX)
ax2.set_ylim(Y_MIN, Y_MAX)
# ---------------------------------------------

# --- Plot IR Transmittance on the left y-axis ---
ax1.set_ylabel('Transmittance (a.u.)', color='red')
ax1.plot(x_axis, ir_y_offset, color='red', lw=1, label='IR (Transmittance)')
ax1.tick_params(axis='y', labelcolor='red')
# CRITICAL FIX: Manually set the ticks and labels for the IR axis
# The data is at 1.2, 1.7, 2.2, but we label it as 0.0, 0.5, 1.0
ax1.set_yticks([1.2, 1.7, 2.2])
ax1.set_yticklabels(['0.0', '0.5', '1.0'])


# --- Plot Raman Intensity on the right y-axis ---
ax2.set_ylabel('Raman Intensity (Normalized)', color='blue')
ax2.plot(x_axis, raman_y_normalized, color='blue', lw=1, label='Raman (Intensity)')
ax2.tick_params(axis='y', labelcolor='blue')
# CRITICAL FIX: Manually set the ticks and labels for the Raman axis
ax2.set_yticks([0.0, 0.5, 1.0])


# --- Configure X-Axes ---
ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
ax1.set_xlim(X_MAX, X_MIN) # Invert x-axis for conventional view

# --- Add and format the top x-axis for eV ---
def cm_to_eV(x):
    return x / 8065.54
def eV_to_cm(x):
    return x * 8065.54

ax_top = ax1.secondary_xaxis('top', functions=(cm_to_eV, eV_to_cm))
ax_top.set_xlabel('Energy (eV)')
ax_top.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))


# --- Final plot adjustments ---
ax1.set_title('Calculated IR and Raman Spectra of Thymol')
# Create a single legend for both plots
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.tight_layout()
plt.savefig('thymol.ir-raman.gas.pdf', bbox_inches='tight', transparent=True)
plt.show()
