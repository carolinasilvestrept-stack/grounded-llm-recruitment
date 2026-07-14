# Robustness Re-analysis (existing N=1,800 dataset)

## 1. Candidate-level analysis (addresses pseudoreplication)

- **role_alignment_index**: N=15 candidates, baseline=23.95, grounded=28.13, diff=4.181, t=9.774, p=1.239e-07, Cohen's dz=2.524
- **supportiveness_index**: N=15 candidates, baseline=95.07, grounded=93.20, diff=-1.867, t=-4.678, p=0.0003555, Cohen's dz=-1.208

## 2. Proportional (relative) counterfactual range

- **role_alignment_index**: baseline range = 5.66 (23.6% of mean); grounded range = 5.98 (21.2% of mean)
- **supportiveness_index**: baseline range = 12.98 (13.7% of mean); grounded range = 16.55 (17.8% of mean)

## 3. Subgroup direction of disparity (role_alignment_index)

### gender_condition
**baseline**:
  - female: mean=24.15, std=16.37, n=900
  - male: mean=23.74, std=16.20, n=900
**grounded**:
  - female: mean=28.44, std=16.13, n=900
  - male: mean=27.82, std=15.93, n=900

### ethnicity_condition
**baseline**:
  - majority: mean=23.92, std=16.29, n=900
  - minority: mean=23.98, std=16.28, n=900
**grounded**:
  - majority: mean=27.98, std=15.99, n=900
  - minority: mean=28.27, std=16.07, n=900

### age_condition
**baseline**:
  - older: mean=24.13, std=16.55, n=900
  - younger: mean=23.76, std=16.01, n=900
**grounded**:
  - older: mean=28.31, std=16.15, n=900
  - younger: mean=27.94, std=15.91, n=900

## 4. Intersectional breakdown (gender x ethnicity x age, role_alignment_index)

**baseline**:
  - female/majority/older: mean=24.27, std=16.76, n=225
  - female/majority/younger: mean=24.03, std=16.10, n=225
  - female/minority/older: mean=24.45, std=16.82, n=225
  - female/minority/younger: mean=23.87, std=15.86, n=225
  - male/majority/older: mean=23.81, std=16.26, n=225
  - male/majority/younger: mean=23.55, std=16.15, n=225
  - male/minority/older: mean=24.01, std=16.47, n=225
  - male/minority/younger: mean=23.58, std=16.03, n=225

**grounded**:
  - female/majority/older: mean=28.61, std=16.15, n=225
  - female/majority/younger: mean=28.14, std=16.08, n=225
  - female/minority/older: mean=28.76, std=16.41, n=225
  - female/minority/younger: mean=28.25, std=15.96, n=225
  - male/majority/older: mean=27.65, std=16.11, n=225
  - male/majority/younger: mean=27.53, std=15.70, n=225
  - male/minority/older: mean=28.23, std=16.01, n=225
  - male/minority/younger: mean=27.85, std=15.99, n=225
