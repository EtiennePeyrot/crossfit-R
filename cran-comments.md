## Test environments

* local Windows 10, R 4.2.1
* win-builder (R-devel, x86_64-w64-mingw32, 2025-11-24)

## R CMD check results

### Local (R 4.2.1, Windows 10)

0 errors | 0 warnings | 1 note

* checking for future file timestamps ... NOTE  
  unable to verify current time  

This NOTE appears to be environment-specific (Windows / local clock);
there are no files with timestamps in the future in the package tarball.

### win-builder (R-devel, Windows Server 2022)

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE  
  New submission.  
  Possibly misspelled words in DESCRIPTION:  
  'Debiased', 'debiased'

These are standard terms in the literature on double/debiased machine learning.
