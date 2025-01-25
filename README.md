# Visual Analytics Project - Fall 2024

This repository contains the **Visual Analytics Project** for Fall 2024. The project involves developing a fully functional visual analytics solution that integrates dimensionality reduction techniques, interactive visualizations, and analytical computations.

In this project, we analyzed **Student Performance**. The goal of the project was to visualize how different factors impact a student's grades.

## Authors

The authors of this project are **Group 6**, consisting of the following students:

- **August Nyheim** - 2181755  
- **Oskar Nesheim** - 2182767  
- **Håkon Refsvik** - 2188403

## Prerequisites

This project requires the use of a virtual environment to run. Follow these steps:

1. **Create the virtual environment**:  
   Run the following command in your terminal:

   ```bash
   python3 -m venv myenv
   ```

2. **Activate the virtual environment**:  
   Activate it by running:

   ```bash
   source myenv/bin/activate
   ```

3. **Install the dependencies**:  
   Install all the required packages listed in `requirements.txt` by running:

   ```bash
   pip install -r requirements.txt
   ```

   This will download and install all necessary packages for the project.

4. **Install additional packages (if needed)**:  
   If you need to install packages not listed in `requirements.txt`, you can do so using:

   ```bash
   pip install <package>
   ```

   Replace `<package>` with the name of the package you want to install.

## How to Run

After installing the necessary prerequisites, the project can be run with the following command:

```bash
python3 vaSystem.py
```

## Attribute Descriptions

### Feature attributes

1. **school**: Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2. **sex**: Student's sex (binary: 'F' - female or 'M' - male)
3. **age**: Student's age (numeric: from 15 to 22)
4. **address**: Student's home address type (binary: 'U' - urban or 'R' - rural)
5. **famsize**: Family size (binary: 'LE3' - less or equal to 3, 'GT3' - greater than 3)
6. **Pstatus**: Parent's cohabitation status (binary: 'T' - living together, 'A' - apart)
7. **Medu**: Mother's education (numeric:  
   - 0: None  
   - 1: Primary education (4th grade)  
   - 2: 5th to 9th grade  
   - 3: Secondary education  
   - 4: Higher education)
8. **Fedu**: Father's education (numeric:  
   - 0: None  
   - 1: Primary education (4th grade)  
   - 2: 5th to 9th grade  
   - 3: Secondary education  
   - 4: Higher education)
9. **Mjob**: Mother's job (nominal: 'teacher', 'health', 'services', 'at_home', 'other')
10. **Fjob**: Father's job (nominal: 'teacher', 'health', 'services', 'at_home', 'other')
11. **reason**: Reason to choose this school (nominal: 'home', 'reputation', 'course', 'other')
12. **guardian**: Student's guardian (nominal: 'mother', 'father', 'other')
13. **traveltime**: Home-to-school travel time (numeric:  
    - 1: <15 min.  
    - 2: 15–30 min.  
    - 3: 30 min.–1 hour  
    - 4: >1 hour)
14. **studytime**: Weekly study time (numeric:  
    - 1: <2 hours  
    - 2: 2–5 hours  
    - 3: 5–10 hours  
    - 4: >10 hours)
15. **failures**: Number of past class failures (numeric: `n` if `1 <= n < 3`, else `4`)
16. **schoolsup**: Extra educational support (binary: 'yes' or 'no')
17. **famsup**: Family educational support (binary: 'yes' or 'no')
18. **paid**: Extra paid classes within the course subject (binary: 'yes' or 'no')
19. **activities**: Extra-curricular activities (binary: 'yes' or 'no')
20. **nursery**: Attended nursery school (binary: 'yes' or 'no')
21. **higher**: Plans for higher education (binary: 'yes' or 'no')
22. **internet**: Internet access at home (binary: 'yes' or 'no')
23. **romantic**: In a romantic relationship (binary: 'yes' or 'no')
24. **famrel**: Quality of family relationships (numeric: 1 - very bad, 5 - excellent)
25. **freetime**: Free time after school (numeric: 1 - very low, 5 - very high)
26. **goout**: Frequency of going out with friends (numeric: 1 - very low, 5 - very high)
27. **Dalc**: Workday alcohol consumption (numeric: 1 - very low, 5 - very high)
28. **Walc**: Weekend alcohol consumption (numeric: 1 - very low, 5 - very high)
29. **health**: Current health status (numeric: 1 - very bad, 5 - very good)
30. **absences**: Number of school absences (numeric: from 0 to 93)

### Course Grades (Specific to Math or Portuguese)
31. **G1**: First period grade (numeric: 0 to 20)
32. **G2**: Second period grade (numeric: 0 to 20)
33. **G3**: Final grade (numeric: 0 to 20) *(Output target)*
