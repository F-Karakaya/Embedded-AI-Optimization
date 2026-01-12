# Power Measurement in Edge AI

Accurately measuring power consumption is critical for battery-powered edge devices. Since this repository is running in a Windows environment, we document the professional methodology used when attached to physical hardware.

## Methodology provided Hardware

### 1. NVIDIA Jetson (Tegra) Family
**Tool:** `tegrastats` (Built-in utility)

**Usage:**
```bash
sudo tegrastats --interval 1000 --logfile power_log.txt
```

**Metrics:**
- `VDD_IN`: Total power input (mW).
- `VDD_CPU_GPU_CV`: Power breakdown for different rails.

**Analysis:**
We run the `inference_trt.py` script in a loop while `tegrastats` records in the background. Post-processing involves parsing the log to calculate average wattage during the active inference window (subtracting idle power baseline).

### 2. Raspberry Pi
**Tool:** USB Power Meter (Hardware) or `vcgencmd` (Voltage only, limited).

**Professional Approach:**
Connect the Pi power supply through a specialized USB-C Power Meter (e.g., UM34C) with Bluetooth logging, or a bench power supply with current monitoring.

**Software Approximation:**
Though less accurate, monitoring CPU load alongside known TDP curves can give rough estimates, but direct current measurement is required for engineering sign-off.

### 3. Latency-Power Tradeoff (The "Efficiency" Metric)
We calculate **Inference Efficiency** as:
$$ \text{FPS per Watt} = \frac{\text{Frames Per Second}}{\text{Average Power (W)}} $$

Or **joules per inference**:
$$ \text{Energy (J)} = \text{Power (W)} \times \text{Latency (s)} $$

## Windows Simulation Note
On Windows Intel/AMD CPUs, we can use tools like `Open Hardware Monitor` or Intel `RAPL` (Running Average Power Limit) interfaces to estimate package power.
For this portfolio, we assume the following nominal TDPs for theoretical comparison:
- **MobileNetV2 (FP32)**: High compute, high energy.
- **MobileNetV2 (INT8)**: Lower memory bandwidth, improved cache hit rate -> reduced energy per inference.
