# Material-property source audit

Reviewed 2026-09-27. This audit covers ordinary density and elastic properties.
All runtime values, material keys, and display names were preserved. A value
falling within a published range is a plausible approximation, not verification
of a particular material grade or of the simulation's contact/failure response.

## Units and meaning

| Field | Units | Interpretation in this project |
| --- | --- | --- |
| `density` | kg/m³ | Nominal bulk density; the code assigns mass using nominal lattice spacing. |
| `youngs_modulus` | Pa | Nominal elastic modulus, subsequently multiplied by numerical scale factors. |
| `poisson_ratio` | Dimensionless | Nominal metadata; currently not consumed by the solver. |
| `hardness` | No consistent scale | Unused legacy ranking. Comments mix Mohs, Vickers, Brinell, Rockwell and Shore values; these are not interchangeable measurements. |
| `strength` | Pa | Legacy contact coefficient. Entries mix tensile, compressive, flow and yield strength, so the field is not a consistent measured property. |
| `restitution`, `friction_coeff` | Dimensionless | Legacy contact settings; measurements depend on the contacting pair and conditions. |
| Spring strain thresholds, `compression_limit`, `spring_min_ratio`, `shear_boost` | Dimensionless | Numerical-model settings. Historical comments describing hardness or modulus ratios do not establish these settings as measured constants. |
| `stiffness_scale` | Physical units not specified | Legacy numerical scaling; this audit does not establish the dimensional consistency of the resulting spring-force calculation. |

For source conversions, 1 g/cm³ = 1,000 kg/m³ and 1 GPa = 10⁹ Pa.
`shear_boost` is a multiplier on selected springs; it is not the shear modulus
or the ratio of shear modulus to Young's modulus.

## Bulk-property review

The stored column lists **density (kg/m³), Young's modulus (GPa), Poisson's
ratio**. Conditions are only asserted where the cited source specifies them.

| Entry | Stored values | Evidence and verification status |
| --- | --- | --- |
| Pure tungsten | 19,300; 411; 0.28 | [Midwest Tungsten][w] reports 19.3 g/cm³ at 20°C and ν=0.284; [Plansee][plansee] reports 19.25 g/cm³ at 20°C. Density and rounded ν are supported. E=411 GPa is corroborated by [a secondary reference][w-secondary], but was not independently verified in a manufacturer/institutional source. |
| Tungsten carbide | 15,000; 630; 0.22 | **Exact values unverified in this review.** The previous comments assumed WC–6Co, while the display name does not specify binder or grade. [CoorsTek][wc] distinguishes bonded and hot-pressed materials and describes nickel/cobalt binders; its page does not verify this density/E/ν combination. |
| Steel core | 7,850; 210; 0.29 | [AZoM AISI 52100][52100] gives 7,810 kg/m³, E=190–210 GPa and ν=0.27–0.30. [Ovako 100Cr6][ovako] gives typical 7,800 kg/m³, 210 GPa and 0.3. The stored E/ν fit these references and density is close; an exact grade/heat-treatment match is not established. |
| Steel, high-hardness steel, ultra-high-hardness steel | 7,850; 207 or 210; 0.29 | [AZoM AISI 4140][4140] gives 7,850 kg/m³, E=190–210 GPa and ν=0.27–0.30. These support generic low-alloy-steel approximations, not verification of every named steel class or its other coefficients. |
| Copper | 8,960; 117; 0.34 | [Copper Development Association C10100][cu] reports specific gravity 8.94 and E=17,000 ksi (about 117.21 GPa). Approximate density and E agree. ν=0.34 was not explicitly verified by this source; the exact temper is unspecified. |
| Alumina | 3,960; 370; 0.22 | [Precision Ceramics][alumina] lists 99.7% alumina at 3,950 kg/m³, 370 GPa, ν=0.23; its 96% and 99.95% grades differ. [AZoM's broad material range][alumina-range] includes the stored values: 3,000–3,980 kg/m³, 215–413 GPa, ν=0.21–0.33. Plausible high-purity values; a single matched grade remains unspecified. |
| Boron carbide | 2,520; 460; 0.17 | [Precision Ceramics][b4c] gives density 2.52 g/cm³. [AZoM][b4c-range] gives E=362–472 GPa and ν=0.18–0.21. Density and E are supported as representative values. **ν=0.17 is outside that reference range and remains unverified**, rather than silently replacing it with a range midpoint. |
| Silicon carbide | 3,210; 410; 0.19 | **A matched grade was not verified.** [Precision Ceramics CeramaSil-C][sic] instead lists 3,100 kg/m³, 350–400 GPa and ν=0.14. This is evidence of grade dependence, not sufficient evidence that the existing unspecified-grade values should be replaced. |
| Soda-lime glass | 2,500; 72; 0.22 | **All three match** [Pilkington ATS-129, September 2026, p. 1][glass]. The property rows do not explicitly specify temperature; the source's temperature labels for other properties should not be transferred to these rows. |
| Ti-6Al-4V | 4,430; 114; 0.34 | [AZoM Grade 5][ti] gives typical 4,420 kg/m³ and 114 GPa. [MakeItFrom's grade summary][ti-range] gives 110 GPa and ν=0.32 across processing variants. Approximate density/E are supported; exact ν=0.34 remains unverified. The old “113–116 GPa across all sources” comment was too strong. |
| Aluminium 5083 | 2,660; 70.3; 0.33 | [AZoM/Aalco][al5083] gives 2,650 kg/m³ and 72 GPa; [MakeItFrom][al5083-range] gives 68 GPa and ν=0.33 across tempers. Approximate density and ν are supported. E=70.3 GPa lies between these reported moduli, but its exact provenance was not established. |
| UHMWPE / Dyneema | 970; 100; 0.42 | [Bally Ribbon Mills][fibre-density] lists approximately 0.97 g/cm³ for Spectra/Dyneema fibre, with manufacturer/yarn-dependent variation. **The exact 100 GPa and ν=0.42 values remain unverified.** The entry combines fibre-inspired values with an isotropic scalar model; it is not a verified bulk-polymer or laminate property set. |

The `depleted_uranium` and `era_filler` entries were not source-verified in this
review. `adamantium` is fictional. None of the strength, fracture, contact,
explosive, or numerical tuning settings were calibrated by this audit.

## Fibre and composite distinction

The Bally Ribbon Mills table labels its value “Specific gravity (g/cc),” mixing
a dimensionless quantity with density units. The approximately 970 kg/m³ fibre
density interpretation above retains that source limitation.

Dyneema's [manufacturing explanation][fibre-process] describes molecular
alignment during fibre production and the separate resin/cross-ply processing
used to make composites. Its [current fibre portfolio][fibre-portfolio] reports
SK99 modulus as **1,590 cN/dtex**, a specific longitudinal tensile modulus. It is
neither a modulus in GPa nor an isotropic bulk-property specification, and it
does not establish the previous SK76-specific attribution. Fibre direction,
resin content and product form must accompany a composite-property reference.

## Source and interpretation limitations

- Broad material ranges contain different grades and conditions. Their endpoints
  should not be assembled into a fictitious single material specification.
- Manufacturer typical values are not guaranteed properties of every component.
- The [live Aalco 5083 O/H111 page][aalco-live] printed “265 g/cm³” when reviewed.
  The AZoM/Aalco reference above explicitly reports 2,650 kg/m³; the anomalous
  live-page entry was not used as a numerical input.
- This audit records uncertainty instead of inventing exact values or changing
  the model to reproduce a desired outcome. The source links apply to the bulk
  properties stated here, not to neighboring solver coefficients in the code.

[w]: https://www.tungsten.com/material-info/tungsten-w
[plansee]: https://www.plansee.com/en/materials/tungsten.html
[w-secondary]: https://en.wikipedia.org/wiki/Tungsten
[wc]: https://www.coorstek.com/en/materials/tungsten-carbide/
[52100]: https://www.azom.com/article.aspx?ArticleID=6704
[ovako]: https://steelnavigator.ovako.com/steel-grades/100cr6/
[4140]: https://www.azom.com/article.aspx?ArticleID=6769
[cu]: https://alloys.copper.org/alloy/C10100
[alumina]: https://precision-ceramics.com/materials/alumina/
[alumina-range]: https://www.azom.com/properties.aspx?ArticleID=52
[b4c]: https://precision-ceramics.com/materials/boron-carbide/
[b4c-range]: https://www.azom.com/properties.aspx?ArticleID=75
[sic]: https://precision-ceramics.com/materials/silicon-carbide/
[glass]: https://www.pilkington.com/-/media/pilkingtonredesign/downloads/usa/architectural/technical-bulletins/ats-129---properties-of-soda-lime-silica-float-glass.pdf
[ti]: https://www.azom.com/article.aspx?ArticleID=1547
[ti-range]: https://www.makeitfrom.com/material-properties/Grade-5-Ti-6Al-4V-3.7165-R56400-Titanium
[al5083]: https://www.azom.com/article.aspx?ArticleID=2804
[al5083-range]: https://www.makeitfrom.com/material-properties/5083-AlMg4.5Mn0.7-3.3547-N8-A95083-Aluminum
[aalco-live]: https://www.aalco.co.uk/datasheets/Aluminium-Alloy-5083-0-H111-Sheet-and-Plate_149.ashx
[fibre-density]: https://www.ballyribbon.com/fibers/spectra-dyneema/
[fibre-process]: https://www.dyneema.com/experience-more/the-science-of-dyneema
[fibre-portfolio]: https://www.dyneema.com/design-with-dyneema/dyneema-product-portfolio
