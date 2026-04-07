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
pub const HBAR: f64 = 1.054_571_817e-34;

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
pub const CONDUCTANCE_QUANTUM: f64 = 7.748_091_729e-5;

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
pub const COMPTON_WAVELENGTH: f64 = 2.426_310_238_67e-12;

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
pub const TORR: f64 = 133.322_368_421_053;
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
}
