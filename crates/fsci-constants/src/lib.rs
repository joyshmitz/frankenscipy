#![forbid(unsafe_code)]

//! Physical and mathematical constants for FrankenSciPy.
//!
//! Matches `scipy.constants` — CODATA 2018 recommended values.
//!
//! # Usage
//! ```
//! use fsci_constants::*;
//! let energy = ELECTRON_MASS * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
//! ```

// ══════════════════════════════════════════════════════════════════════
// Mathematical Constants
// ══════════════════════════════════════════════════════════════════════

/// π
pub const PI: f64 = std::f64::consts::PI;
/// 2π (tau)
pub const TAU: f64 = 2.0 * std::f64::consts::PI;
/// Euler's number e
pub const E: f64 = std::f64::consts::E;
/// Golden ratio φ = (1 + √5)/2
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
/// Euler-Mascheroni constant γ
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ══════════════════════════════════════════════════════════════════════
// SI Prefixes
// ══════════════════════════════════════════════════════════════════════

pub const YOTTA: f64 = 1e24;
pub const ZETTA: f64 = 1e21;
pub const EXA: f64 = 1e18;
pub const PETA: f64 = 1e15;
pub const TERA: f64 = 1e12;
pub const GIGA: f64 = 1e9;
pub const MEGA: f64 = 1e6;
pub const KILO: f64 = 1e3;
pub const HECTO: f64 = 1e2;
pub const DEKA: f64 = 1e1;
pub const DECI: f64 = 1e-1;
pub const CENTI: f64 = 1e-2;
pub const MILLI: f64 = 1e-3;
pub const MICRO: f64 = 1e-6;
pub const NANO: f64 = 1e-9;
pub const PICO: f64 = 1e-12;
pub const FEMTO: f64 = 1e-15;
pub const ATTO: f64 = 1e-18;
pub const ZEPTO: f64 = 1e-21;
pub const YOCTO: f64 = 1e-24;

// ══════════════════════════════════════════════════════════════════════
// Fundamental Physical Constants (CODATA 2018)
// ══════════════════════════════════════════════════════════════════════

/// Speed of light in vacuum [m/s]
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Shorthand alias
pub const C: f64 = SPEED_OF_LIGHT;

/// Magnetic constant (vacuum permeability) μ₀ [N/A²]
pub const MU_0: f64 = 1.256_637_062_12e-6;

/// Electric constant (vacuum permittivity) ε₀ [F/m]
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// Planck constant [J·s]
pub const PLANCK: f64 = 6.626_070_15e-34;
/// Shorthand alias
pub const H: f64 = PLANCK;

/// Reduced Planck constant ℏ = h/(2π) [J·s]
///
/// Defined as `PLANCK / TAU` so precision is automatically correct to
/// machine epsilon. The prior hand-typed `1.054_571_817e-34` truncated
/// 4-5 significant digits vs scipy.constants.hbar (6.13e-10 relative
/// error). Per frankenscipy-h0su.
pub const HBAR: f64 = PLANCK / TAU;

/// Newtonian constant of gravitation [m³/(kg·s²)]
pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11;
/// Shorthand alias
pub const G: f64 = GRAVITATIONAL_CONSTANT;

/// Standard acceleration of gravity [m/s²]
pub const G_N: f64 = 9.806_65;

/// Elementary charge [C]
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Shorthand alias
pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;

/// Molar gas constant R [J/(mol·K)]
pub const GAS_CONSTANT: f64 = 8.314_462_618;
/// Shorthand alias
pub const R: f64 = GAS_CONSTANT;

/// Avogadro constant [1/mol]
pub const AVOGADRO: f64 = 6.022_140_76e23;
/// Shorthand alias
pub const N_A: f64 = AVOGADRO;

/// Boltzmann constant [J/K]
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Shorthand alias
pub const K_B: f64 = BOLTZMANN;

/// Stefan-Boltzmann constant [W/(m²·K⁴)]
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Shorthand alias
pub const SIGMA: f64 = STEFAN_BOLTZMANN;

/// Wien displacement law constant [m·K]
pub const WIEN: f64 = 2.897_771_955e-3;

/// Rydberg constant [1/m]
pub const RYDBERG: f64 = 1.097_373_156_816_0e7;

// ══════════════════════════════════════════════════════════════════════
// Particle Masses
// ══════════════════════════════════════════════════════════════════════

/// Electron mass [kg]
pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;
/// Shorthand alias
pub const M_E: f64 = ELECTRON_MASS;

/// Proton mass [kg]
pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;
/// Shorthand alias
pub const M_P: f64 = PROTON_MASS;

/// Neutron mass [kg]
pub const NEUTRON_MASS: f64 = 1.674_927_498_04e-27;
/// Shorthand alias
pub const M_N: f64 = NEUTRON_MASS;

/// Atomic mass constant (1/12 of ¹²C mass) [kg]
pub const ATOMIC_MASS: f64 = 1.660_539_066_60e-27;
/// Shorthand alias
pub const U: f64 = ATOMIC_MASS;

// ══════════════════════════════════════════════════════════════════════
// Electromagnetic Constants
// ══════════════════════════════════════════════════════════════════════

/// Bohr magneton [J/T]
pub const BOHR_MAGNETON: f64 = 9.274_010_078_3e-24;

/// Nuclear magneton [J/T]
pub const NUCLEAR_MAGNETON: f64 = 5.050_783_746_1e-27;

/// Magnetic flux quantum Φ₀ = h/(2e) [Wb]
pub const MAGNETIC_FLUX_QUANTUM: f64 = 2.067_833_848e-15;

/// Conductance quantum G₀ = 2e²/h [S]
///
/// Derived as `2 * ELEMENTARY_CHARGE^2 / PLANCK` so precision tracks the
/// SI-exact inputs. Per frankenscipy-h0su.
pub const CONDUCTANCE_QUANTUM: f64 = 2.0 * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE / PLANCK;

/// Josephson constant K_J = 2e/h [Hz/V]
pub const JOSEPHSON: f64 = 4.835_978_484e14;

/// Von Klitzing constant R_K = h/e² [Ω]
pub const VON_KLITZING: f64 = 2.581_280_745e4;

// ══════════════════════════════════════════════════════════════════════
// Atomic & Nuclear Constants
// ══════════════════════════════════════════════════════════════════════

/// Fine-structure constant α
pub const FINE_STRUCTURE: f64 = 7.297_352_569_3e-3;
/// Shorthand alias
pub const ALPHA: f64 = FINE_STRUCTURE;

/// Bohr radius [m]
pub const BOHR_RADIUS: f64 = 5.291_772_109_03e-11;

/// Hartree energy [J]
pub const HARTREE: f64 = 4.359_744_722_207_1e-18;

/// Classical electron radius [m]
pub const CLASSICAL_ELECTRON_RADIUS: f64 = 2.817_940_326_2e-15;

/// Compton wavelength of electron [m]
pub const COMPTON_WAVELENGTH: f64 = 2.426_310_235_38e-12;
/// Explicit electron alias for the physical_constants table key.
pub const ELECTRON_COMPTON_WAVELENGTH: f64 = COMPTON_WAVELENGTH;

// ══════════════════════════════════════════════════════════════════════
// Additional Physical Constants — br-wada
// ══════════════════════════════════════════════════════════════════════
//
// CODATA 2022 / scipy.constants.physical_constants. Bit-exact parity
// against scipy is locked in via FSCI-P2C-016 fixture cases.

/// Faraday constant F = N_A · e [C·mol⁻¹]
pub const FARADAY: f64 = 96_485.332_123_310_01;

/// Electron g-factor (dimensionless, negative)
pub const ELECTRON_G_FACTOR: f64 = -2.002_319_304_360_92;

/// Proton g-factor (dimensionless)
pub const PROTON_G_FACTOR: f64 = 5.585_694_689_3;

/// Neutron g-factor (dimensionless, negative)
pub const NEUTRON_G_FACTOR: f64 = -3.826_085_52;

/// Muon g-factor (dimensionless, negative)
pub const MUON_G_FACTOR: f64 = -2.002_331_841_23;

/// Thomson cross section [m²]
pub const THOMSON_CROSS_SECTION: f64 = 6.652_458_705_1e-29;

/// Characteristic impedance of vacuum [Ω]
pub const CHARACTERISTIC_IMPEDANCE_OF_VACUUM: f64 = 376.730_313_412;

/// Deuteron mass [kg]
pub const DEUTERON_MASS: f64 = 3.343_583_776_8e-27;

/// Alpha particle mass [kg]
pub const ALPHA_PARTICLE_MASS: f64 = 6.644_657_345e-27;

/// Muon mass [kg]
pub const MUON_MASS: f64 = 1.883_531_627e-28;

/// Tau mass [kg]
pub const TAU_MASS: f64 = 3.167_54e-27;

/// Helion mass [kg]
pub const HELION_MASS: f64 = 5.006_412_786_2e-27;

/// Triton mass [kg]
pub const TRITON_MASS: f64 = 5.007_356_751_2e-27;

/// Molar volume of ideal gas (273.15 K, 101.325 kPa) [m³·mol⁻¹]
pub const MOLAR_VOLUME_IDEAL_GAS: f64 = 0.022_413_969_545_014_137;

/// Molar Planck constant [J·Hz⁻¹·mol⁻¹]
pub const MOLAR_PLANCK: f64 = 3.990_312_712_893_431e-10;

/// Rydberg constant times c in Hz [Hz]
pub const RYDBERG_HZ: f64 = 3_289_841_960_250_000.0;

/// Inverse fine-structure constant α⁻¹ (dimensionless)
pub const INVERSE_FINE_STRUCTURE: f64 = 137.035_999_177;

/// First radiation constant c₁ = 2π·h·c² [W·m²]
pub const FIRST_RADIATION_CONSTANT: f64 = 3.741_771_852_192_757_3e-16;

/// Second radiation constant c₂ = h·c/k [m·K]
pub const SECOND_RADIATION_CONSTANT: f64 = 0.014_387_768_775_039_337;

/// Electron-proton mass ratio (dimensionless)
pub const ELECTRON_PROTON_MASS_RATIO: f64 = 0.000_544_617_021_488_9;

/// Proton-electron mass ratio (dimensionless)
pub const PROTON_ELECTRON_MASS_RATIO: f64 = 1_836.152_673_426;

/// Bohr magneton in eV/T [eV·T⁻¹]
pub const BOHR_MAGNETON_EV_T: f64 = 5.788_381_798_2e-5;

// ══════════════════════════════════════════════════════════════════════
// Additional Physical Constants — br-sl60
// ══════════════════════════════════════════════════════════════════════

/// Electron mass energy equivalent [MeV]
pub const ELECTRON_MASS_MEV: f64 = 0.510_998_950_69;

/// Proton mass energy equivalent [MeV]
pub const PROTON_MASS_MEV: f64 = 938.272_089_43;

/// Neutron mass energy equivalent [MeV]
pub const NEUTRON_MASS_MEV: f64 = 939.565_421_94;

/// Muon mass energy equivalent [MeV]
pub const MUON_MASS_MEV: f64 = 105.658_375_5;

/// Tau mass energy equivalent [MeV]
pub const TAU_MASS_MEV: f64 = 1_776.82;

/// Deuteron mass energy equivalent [MeV]
pub const DEUTERON_MASS_MEV: f64 = 1_875.612_945;

/// Alpha particle mass energy equivalent [MeV]
pub const ALPHA_PARTICLE_MASS_MEV: f64 = 3_727.379_411_8;

/// Helion mass energy equivalent [MeV]
pub const HELION_MASS_MEV: f64 = 2_808.391_611_12;

/// Triton mass energy equivalent [MeV]
pub const TRITON_MASS_MEV: f64 = 2_808.921_136_68;

/// Proton Compton wavelength [m]
pub const PROTON_COMPTON_WAVELENGTH: f64 = 1.321_409_853_6e-15;

/// Neutron Compton wavelength [m]
pub const NEUTRON_COMPTON_WAVELENGTH: f64 = 1.319_590_903_82e-15;

/// Muon Compton wavelength [m]
pub const MUON_COMPTON_WAVELENGTH: f64 = 1.173_444_11e-14;

/// Tau Compton wavelength [m]
pub const TAU_COMPTON_WAVELENGTH: f64 = 6.977_71e-16;

/// Neutron-electron mass ratio (dimensionless)
pub const NEUTRON_ELECTRON_MASS_RATIO: f64 = 1_838.683_662;

/// Electron-neutron mass ratio (dimensionless)
pub const ELECTRON_NEUTRON_MASS_RATIO: f64 = 0.000_543_867_344_16;

/// Muon-electron mass ratio (dimensionless)
pub const MUON_ELECTRON_MASS_RATIO: f64 = 206.768_282_7;

/// Electron-muon mass ratio (dimensionless)
pub const ELECTRON_MUON_MASS_RATIO: f64 = 0.004_836_331_7;

/// Proton-neutron mass ratio (dimensionless)
pub const PROTON_NEUTRON_MASS_RATIO: f64 = 0.998_623_477_97;

/// Neutron-proton mass ratio (dimensionless)
pub const NEUTRON_PROTON_MASS_RATIO: f64 = 1.001_378_419_46;

/// Deuteron-electron mass ratio (dimensionless)
pub const DEUTERON_ELECTRON_MASS_RATIO: f64 = 3_670.482_967_655;

/// Alpha particle-electron mass ratio (dimensionless)
pub const ALPHA_PARTICLE_ELECTRON_MASS_RATIO: f64 = 7_294.299_541_71;

/// Helion-electron mass ratio (dimensionless)
pub const HELION_ELECTRON_MASS_RATIO: f64 = 5_495.885_279_84;

/// Triton-electron mass ratio (dimensionless)
pub const TRITON_ELECTRON_MASS_RATIO: f64 = 5_496.921_535_51;

/// Tau-electron mass ratio (dimensionless)
pub const TAU_ELECTRON_MASS_RATIO: f64 = 3_477.23;

/// Electron-tau mass ratio (dimensionless)
pub const ELECTRON_TAU_MASS_RATIO: f64 = 0.000_287_585;

// ══════════════════════════════════════════════════════════════════════
// Conversion Factors
// ══════════════════════════════════════════════════════════════════════

/// Electron volt [J]
pub const ELECTRON_VOLT: f64 = 1.602_176_634e-19;
/// Shorthand alias
pub const EV: f64 = ELECTRON_VOLT;

/// Calorie (thermochemical) [J]
pub const CALORIE: f64 = 4.184;

/// Standard atmosphere [Pa]
pub const ATMOSPHERE: f64 = 101_325.0;
/// Shorthand alias
pub const ATM: f64 = ATMOSPHERE;

/// Torr (mmHg) [Pa]
///
/// Defined as `ATMOSPHERE / 760.0` — the physics-exact ratio.
/// Per frankenscipy-h0su.
pub const TORR: f64 = ATMOSPHERE / 760.0;
/// Shorthand alias
pub const MMHG: f64 = TORR;

/// Bar [Pa]
pub const BAR: f64 = 1e5;

/// Pound-force [N]
pub const POUND_FORCE: f64 = 4.448_221_615_260_5;

/// Angstrom [m]
pub const ANGSTROM: f64 = 1e-10;

/// Nautical mile [m]
pub const NAUTICAL_MILE: f64 = 1852.0;

/// Light year [m]
pub const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;

/// Astronomical unit [m]
pub const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;
/// Shorthand alias
pub const AU: f64 = ASTRONOMICAL_UNIT;

/// Parsec [m]
pub const PARSEC: f64 = 3.085_677_581_28e16;

/// Inch [m]
pub const INCH: f64 = 0.0254;

/// Foot [m]
pub const FOOT: f64 = 0.3048;

/// Yard [m]
pub const YARD: f64 = 0.9144;

/// Mile [m]
pub const MILE: f64 = 1609.344;

/// Pound mass (avoirdupois) [kg]
pub const POUND: f64 = 0.453_592_37;

/// Ounce (avoirdupois) [kg]
pub const OUNCE: f64 = 0.028_349_523_125;

/// Gallon (US liquid) [m³]
pub const GALLON: f64 = 3.785_411_784e-3;

/// Liter [m³]
pub const LITER: f64 = 1e-3;

/// Degree (angle) [rad]
pub const DEGREE: f64 = std::f64::consts::PI / 180.0;

/// Arc minute [rad]
pub const ARCMINUTE: f64 = DEGREE / 60.0;

/// Arc second [rad]
pub const ARCSECOND: f64 = DEGREE / 3600.0;

// ══════════════════════════════════════════════════════════════════════
// Temperature Conversion Functions
// ══════════════════════════════════════════════════════════════════════

/// Convert Celsius to Kelvin.
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Convert Kelvin to Celsius.
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Convert Fahrenheit to Kelvin.
pub fn fahrenheit_to_kelvin(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0 + 273.15
}

/// Convert Kelvin to Fahrenheit.
pub fn kelvin_to_fahrenheit(k: f64) -> f64 {
    (k - 273.15) * 9.0 / 5.0 + 32.0
}

/// Convert Fahrenheit to Celsius.
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Convert Celsius to Fahrenheit.
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Convert Rankine to Kelvin.
pub fn rankine_to_kelvin(r: f64) -> f64 {
    r * 5.0 / 9.0
}

/// Convert Kelvin to Rankine.
pub fn kelvin_to_rankine(k: f64) -> f64 {
    k * 9.0 / 5.0
}

// ══════════════════════════════════════════════════════════════════════
// Energy Conversion
// ══════════════════════════════════════════════════════════════════════

/// Convert electron volts to Joules.
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * ELECTRON_VOLT
}

/// Convert Joules to electron volts.
pub fn joules_to_ev(j: f64) -> f64 {
    j / ELECTRON_VOLT
}

/// Convert wavelength [m] to frequency [Hz].
pub fn wavelength_to_freq(wavelength: f64) -> f64 {
    SPEED_OF_LIGHT / wavelength
}

/// Convert frequency [Hz] to wavelength [m].
pub fn freq_to_wavelength(freq: f64) -> f64 {
    SPEED_OF_LIGHT / freq
}

/// Lookup a physical constant by name (case-insensitive).
///
/// Matches `scipy.constants.value(name)`.
pub fn value(name: &str) -> Option<f64> {
    match name.to_lowercase().as_str() {
        "speed of light" | "c" => Some(SPEED_OF_LIGHT),
        "planck" | "h" => Some(PLANCK),
        "hbar" => Some(HBAR),
        "gravitational constant" | "g" => Some(GRAVITATIONAL_CONSTANT),
        "elementary charge" | "e" => Some(ELEMENTARY_CHARGE),
        "gas constant" | "r" => Some(GAS_CONSTANT),
        "avogadro" | "n_a" => Some(AVOGADRO),
        "boltzmann" | "k" | "k_b" => Some(BOLTZMANN),
        "stefan-boltzmann" | "sigma" => Some(STEFAN_BOLTZMANN),
        "wien" => Some(WIEN),
        "rydberg" => Some(RYDBERG),
        "electron mass" | "m_e" => Some(ELECTRON_MASS),
        "fine-structure" | "alpha" => Some(FINE_STRUCTURE),
        "bohr radius" => Some(BOHR_RADIUS),
        "electron volt" | "ev" => Some(ELECTRON_VOLT),
        "atmosphere" | "atm" => Some(ATMOSPHERE),
        "proton mass" | "m_p" => Some(PROTON_MASS),
        "neutron mass" | "m_n" => Some(NEUTRON_MASS),
        "atomic mass" | "u" => Some(ATOMIC_MASS),
        "mu_0" | "magnetic constant" => Some(MU_0),
        "epsilon_0" | "electric constant" => Some(EPSILON_0),
        // br-wada additions — keys mirror scipy.constants.physical_constants
        // exactly (case-insensitive) so value() is a drop-in for scipy's.
        "faraday constant" | "faraday" => Some(FARADAY),
        "electron g factor" => Some(ELECTRON_G_FACTOR),
        "proton g factor" => Some(PROTON_G_FACTOR),
        "neutron g factor" => Some(NEUTRON_G_FACTOR),
        "muon g factor" => Some(MUON_G_FACTOR),
        "thomson cross section" => Some(THOMSON_CROSS_SECTION),
        "characteristic impedance of vacuum" => Some(CHARACTERISTIC_IMPEDANCE_OF_VACUUM),
        "deuteron mass" => Some(DEUTERON_MASS),
        "alpha particle mass" => Some(ALPHA_PARTICLE_MASS),
        "muon mass" => Some(MUON_MASS),
        "tau mass" => Some(TAU_MASS),
        "helion mass" => Some(HELION_MASS),
        "triton mass" => Some(TRITON_MASS),
        "molar volume of ideal gas (273.15 k, 101.325 kpa)" => Some(MOLAR_VOLUME_IDEAL_GAS),
        "molar planck constant" => Some(MOLAR_PLANCK),
        "rydberg constant times c in hz" => Some(RYDBERG_HZ),
        "inverse fine-structure constant" => Some(INVERSE_FINE_STRUCTURE),
        "first radiation constant" => Some(FIRST_RADIATION_CONSTANT),
        "second radiation constant" => Some(SECOND_RADIATION_CONSTANT),
        "electron-proton mass ratio" => Some(ELECTRON_PROTON_MASS_RATIO),
        "proton-electron mass ratio" => Some(PROTON_ELECTRON_MASS_RATIO),
        "bohr magneton in ev/t" => Some(BOHR_MAGNETON_EV_T),
        "compton wavelength" => Some(COMPTON_WAVELENGTH),
        "electron mass energy equivalent in mev" => Some(ELECTRON_MASS_MEV),
        "proton mass energy equivalent in mev" => Some(PROTON_MASS_MEV),
        "neutron mass energy equivalent in mev" => Some(NEUTRON_MASS_MEV),
        "muon mass energy equivalent in mev" => Some(MUON_MASS_MEV),
        "tau mass energy equivalent in mev" => Some(TAU_MASS_MEV),
        "deuteron mass energy equivalent in mev" => Some(DEUTERON_MASS_MEV),
        "alpha particle mass energy equivalent in mev" => Some(ALPHA_PARTICLE_MASS_MEV),
        "helion mass energy equivalent in mev" => Some(HELION_MASS_MEV),
        "triton mass energy equivalent in mev" => Some(TRITON_MASS_MEV),
        "proton compton wavelength" => Some(PROTON_COMPTON_WAVELENGTH),
        "neutron compton wavelength" => Some(NEUTRON_COMPTON_WAVELENGTH),
        "muon compton wavelength" => Some(MUON_COMPTON_WAVELENGTH),
        "tau compton wavelength" => Some(TAU_COMPTON_WAVELENGTH),
        "neutron-electron mass ratio" => Some(NEUTRON_ELECTRON_MASS_RATIO),
        "electron-neutron mass ratio" => Some(ELECTRON_NEUTRON_MASS_RATIO),
        "muon-electron mass ratio" => Some(MUON_ELECTRON_MASS_RATIO),
        "electron-muon mass ratio" => Some(ELECTRON_MUON_MASS_RATIO),
        "proton-neutron mass ratio" => Some(PROTON_NEUTRON_MASS_RATIO),
        "neutron-proton mass ratio" => Some(NEUTRON_PROTON_MASS_RATIO),
        "deuteron-electron mass ratio" => Some(DEUTERON_ELECTRON_MASS_RATIO),
        "alpha particle-electron mass ratio" => Some(ALPHA_PARTICLE_ELECTRON_MASS_RATIO),
        "helion-electron mass ratio" => Some(HELION_ELECTRON_MASS_RATIO),
        "triton-electron mass ratio" => Some(TRITON_ELECTRON_MASS_RATIO),
        "tau-electron mass ratio" => Some(TAU_ELECTRON_MASS_RATIO),
        "electron-tau mass ratio" => Some(ELECTRON_TAU_MASS_RATIO),
        _ => None,
    }
}

/// Find a physical constant by partial name match.
///
/// Returns all constants whose name contains the query string.
/// Matches `scipy.constants.find(query)`.
pub fn find(query: &str) -> Vec<(&'static str, f64)> {
    let q = query.to_lowercase();
    let all = [
        ("speed of light in vacuum", SPEED_OF_LIGHT),
        ("Planck constant", PLANCK),
        ("reduced Planck constant", HBAR),
        ("Newtonian constant of gravitation", GRAVITATIONAL_CONSTANT),
        ("standard acceleration of gravity", G_N),
        ("elementary charge", ELEMENTARY_CHARGE),
        ("molar gas constant", GAS_CONSTANT),
        ("Avogadro constant", AVOGADRO),
        ("Boltzmann constant", BOLTZMANN),
        ("Stefan-Boltzmann constant", STEFAN_BOLTZMANN),
        ("Wien displacement law constant", WIEN),
        ("Rydberg constant", RYDBERG),
        ("electron mass", ELECTRON_MASS),
        ("proton mass", PROTON_MASS),
        ("neutron mass", NEUTRON_MASS),
        ("atomic mass constant", ATOMIC_MASS),
        ("Bohr magneton", BOHR_MAGNETON),
        ("nuclear magneton", NUCLEAR_MAGNETON),
        ("fine-structure constant", FINE_STRUCTURE),
        ("Bohr radius", BOHR_RADIUS),
        ("Hartree energy", HARTREE),
        ("electron volt", ELECTRON_VOLT),
        ("standard atmosphere", ATMOSPHERE),
        ("magnetic constant", MU_0),
        ("electric constant", EPSILON_0),
    ];
    all.iter()
        .filter(|(name, _)| name.to_lowercase().contains(&q))
        .cloned()
        .collect()
}

/// Convert between physical units.
///
/// Matches `scipy.constants.convert_temperature` (generalized).
pub fn convert_temperature(val: f64, from: &str, to: &str) -> Result<f64, String> {
    // Convert to Kelvin first
    let kelvin = match from.to_lowercase().as_str() {
        "c" | "celsius" => celsius_to_kelvin(val),
        "f" | "fahrenheit" => fahrenheit_to_kelvin(val),
        "k" | "kelvin" => val,
        "r" | "rankine" => rankine_to_kelvin(val),
        s => return Err(format!("unsupported temperature scale: {}", s)),
    };
    // Convert from Kelvin to target
    match to.to_lowercase().as_str() {
        "c" | "celsius" => Ok(kelvin_to_celsius(kelvin)),
        "f" | "fahrenheit" => Ok(kelvin_to_fahrenheit(kelvin)),
        "k" | "kelvin" => Ok(kelvin),
        "r" | "rankine" => Ok(kelvin_to_rankine(kelvin)),
        s => Err(format!("unsupported temperature scale: {}", s)),
    }
}

/// Convert angle from degrees to radians.
pub fn deg2rad(degrees: f64) -> f64 {
    degrees * DEGREE
}

/// Convert angle from radians to degrees.
pub fn rad2deg(radians: f64) -> f64 {
    radians / DEGREE
}

/// Convert speed from mph to m/s.
pub fn mph_to_mps(mph: f64) -> f64 {
    mph * MILE / 3600.0
}

/// Convert speed from km/h to m/s.
pub fn kmh_to_mps(kmh: f64) -> f64 {
    kmh * 1000.0 / 3600.0
}

/// Convert speed from knots to m/s.
pub fn knots_to_mps(knots: f64) -> f64 {
    knots * NAUTICAL_MILE / 3600.0
}

/// Convert pressure from psi to Pa.
pub fn psi_to_pa(psi: f64) -> f64 {
    psi * POUND_FORCE / (INCH * INCH)
}

/// Convert mass from kg to lb.
pub fn kg_to_lb(kg: f64) -> f64 {
    kg / POUND
}

/// Convert mass from lb to kg.
pub fn lb_to_kg(lb: f64) -> f64 {
    lb * POUND
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64, relation: &str) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "{relation}: actual={actual:.17e}, expected={expected:.17e}, delta={delta:.17e}, tolerance={tolerance:.17e}"
        );
    }

    #[test]
    fn speed_of_light_exact() {
        assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
    }

    #[test]
    fn planck_order_of_magnitude() {
        const {
            assert!(PLANCK > 6e-34 && PLANCK < 7e-34);
        }
    }

    #[test]
    fn temperature_conversions() {
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((celsius_to_kelvin(100.0) - 373.15).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(212.0) - 100.0).abs() < 1e-10);
        assert!((kelvin_to_fahrenheit(0.0) - (-459.67)).abs() < 0.01);
    }

    #[test]
    fn energy_conversions() {
        let j = ev_to_joules(1.0);
        assert!((j - ELECTRON_VOLT).abs() < 1e-30);
        assert!((joules_to_ev(j) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn constant_lookup() {
        assert_eq!(value("speed of light"), Some(SPEED_OF_LIGHT));
        assert_eq!(value("Planck"), Some(PLANCK));
        assert_eq!(value("nonexistent"), None);
    }

    #[test]
    fn golden_ratio_value() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((GOLDEN_RATIO - phi).abs() < 1e-15);
    }

    #[test]
    fn si_prefixes() {
        assert_eq!(KILO, 1e3);
        assert_eq!(MILLI, 1e-3);
        assert_eq!(NANO, 1e-9);
    }

    #[test]
    fn metamorphic_derived_constants_are_self_consistent() {
        let gas_from_micro_constants = BOLTZMANN * AVOGADRO;
        assert!(
            (gas_from_micro_constants - GAS_CONSTANT).abs() < 1e-9,
            "k_B * N_A should recover R"
        );

        let faraday_from_charge_count = ELEMENTARY_CHARGE * AVOGADRO;
        assert!(
            (faraday_from_charge_count - FARADAY).abs() < 1e-9,
            "e * N_A should recover Faraday"
        );

        assert!(
            (HBAR * TAU - PLANCK).abs() < 1e-48,
            "hbar * tau should recover h"
        );

        assert!(
            (CONDUCTANCE_QUANTUM * VON_KLITZING - 2.0).abs() < 1e-9,
            "G0 * R_K should equal 2"
        );
    }

    #[test]
    fn metamorphic_temperature_conversions_are_roundtrip_stable() {
        let kelvin_cases = [0.0, 1.0, 273.15, 310.15, 373.15, 1_000.0];
        let scales = ["kelvin", "celsius", "fahrenheit", "rankine"];

        for kelvin in kelvin_cases {
            for from_scale in scales {
                let source = convert_temperature(kelvin, "kelvin", from_scale)
                    .expect("kelvin should convert to every supported scale");
                for to_scale in scales {
                    let converted = convert_temperature(source, from_scale, to_scale)
                        .expect("supported source and target scales should convert");
                    let roundtrip = convert_temperature(converted, to_scale, from_scale)
                        .expect("supported target scale should convert back");
                    assert_close(
                        roundtrip,
                        source,
                        1e-10,
                        "temperature conversion should roundtrip through any supported scale",
                    );
                }
            }
        }
    }

    #[test]
    fn metamorphic_unit_conversions_are_inverse_and_linear() {
        for ev in [1e-9, 1.0, 13.605_693_122_994, 1e9] {
            assert_close(
                joules_to_ev(ev_to_joules(ev)),
                ev,
                ev.abs().max(1.0) * 1e-15,
                "electron-volt conversion should invert joule conversion",
            );
            assert_close(
                ev_to_joules(ev * 2.0),
                ev_to_joules(ev) * 2.0,
                ev_to_joules(ev.abs()).max(1.0) * 1e-15,
                "electron-volt conversion should be linear",
            );
        }

        for degrees in [-720.0, -45.0, 0.0, 90.0, 360.0] {
            assert_close(
                rad2deg(deg2rad(degrees)),
                degrees,
                1e-12,
                "degree/radian conversion should roundtrip",
            );
        }

        for pounds in [0.0, 1.0, 2.2, 150.5] {
            assert_close(
                kg_to_lb(lb_to_kg(pounds)),
                pounds,
                1e-12,
                "pound/kilogram conversion should roundtrip",
            );
        }

        for wavelength in [1e-12, 532e-9, 21.106_114_054_160e-2] {
            assert_close(
                freq_to_wavelength(wavelength_to_freq(wavelength)),
                wavelength,
                wavelength * 1e-15,
                "wavelength/frequency conversion should roundtrip",
            );
        }
    }

    #[test]
    fn metamorphic_particle_energy_and_wavelength_constants_are_self_consistent() {
        let mass_energy_cases = [
            (
                ELECTRON_MASS,
                ELECTRON_MASS_MEV,
                5e-9,
                "electron mass energy",
            ),
            (PROTON_MASS, PROTON_MASS_MEV, 5e-9, "proton mass energy"),
            (NEUTRON_MASS, NEUTRON_MASS_MEV, 5e-9, "neutron mass energy"),
            (MUON_MASS, MUON_MASS_MEV, 5e-9, "muon mass energy"),
            (TAU_MASS, TAU_MASS_MEV, 5e-5, "tau mass energy"),
            (
                DEUTERON_MASS,
                DEUTERON_MASS_MEV,
                5e-9,
                "deuteron mass energy",
            ),
            (
                ALPHA_PARTICLE_MASS,
                ALPHA_PARTICLE_MASS_MEV,
                5e-9,
                "alpha particle mass energy",
            ),
            (HELION_MASS, HELION_MASS_MEV, 5e-9, "helion mass energy"),
            (TRITON_MASS, TRITON_MASS_MEV, 5e-9, "triton mass energy"),
        ];

        for (mass_kg, energy_mev, relative_tolerance, relation) in mass_energy_cases {
            let derived_mev = joules_to_ev(mass_kg * SPEED_OF_LIGHT * SPEED_OF_LIGHT) / 1e6;
            assert_close(
                derived_mev,
                energy_mev,
                energy_mev.abs().max(1.0) * relative_tolerance,
                relation,
            );
        }

        let compton_cases = [
            (
                ELECTRON_MASS,
                COMPTON_WAVELENGTH,
                5e-9,
                "electron Compton wavelength",
            ),
            (
                PROTON_MASS,
                PROTON_COMPTON_WAVELENGTH,
                5e-9,
                "proton Compton wavelength",
            ),
            (
                NEUTRON_MASS,
                NEUTRON_COMPTON_WAVELENGTH,
                5e-9,
                "neutron Compton wavelength",
            ),
            (
                MUON_MASS,
                MUON_COMPTON_WAVELENGTH,
                5e-9,
                "muon Compton wavelength",
            ),
            (
                TAU_MASS,
                TAU_COMPTON_WAVELENGTH,
                5e-5,
                "tau Compton wavelength",
            ),
        ];

        for (mass_kg, wavelength, relative_tolerance, relation) in compton_cases {
            let derived = PLANCK / (mass_kg * SPEED_OF_LIGHT);
            assert_close(
                derived,
                wavelength,
                wavelength * relative_tolerance,
                relation,
            );
        }
    }

    #[test]
    fn metamorphic_reciprocal_and_radiation_constants_are_self_consistent() {
        let reciprocal_cases = [
            (
                ELECTRON_PROTON_MASS_RATIO,
                PROTON_ELECTRON_MASS_RATIO,
                "electron/proton mass ratios",
            ),
            (
                ELECTRON_NEUTRON_MASS_RATIO,
                NEUTRON_ELECTRON_MASS_RATIO,
                "electron/neutron mass ratios",
            ),
            (
                ELECTRON_MUON_MASS_RATIO,
                MUON_ELECTRON_MASS_RATIO,
                "electron/muon mass ratios",
            ),
            (
                PROTON_NEUTRON_MASS_RATIO,
                NEUTRON_PROTON_MASS_RATIO,
                "proton/neutron mass ratios",
            ),
        ];
        for (forward, inverse, relation) in reciprocal_cases {
            assert_close(forward * inverse, 1.0, 5e-9, relation);
        }

        assert_close(
            PLANCK * AVOGADRO,
            MOLAR_PLANCK,
            MOLAR_PLANCK * 1e-15,
            "molar Planck constant should equal h * N_A",
        );
        assert_close(
            2.0 * PI * PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT,
            FIRST_RADIATION_CONSTANT,
            FIRST_RADIATION_CONSTANT * 1e-15,
            "first radiation constant should equal 2*pi*h*c^2",
        );
        assert_close(
            PLANCK * SPEED_OF_LIGHT / BOLTZMANN,
            SECOND_RADIATION_CONSTANT,
            SECOND_RADIATION_CONSTANT * 1e-15,
            "second radiation constant should equal h*c/k",
        );
    }
}
