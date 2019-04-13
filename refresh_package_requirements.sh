# To be used by developers - whenever a new package dependency is added
conda list --explicit > conda-spec-file.txt
pip freeze > requirements.txt