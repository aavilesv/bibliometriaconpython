import matplotlib.pyplot as plt

# Data for Scopus
scopus_years = [
    2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 
    2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004
]
scopus_counts = [
    34, 31, 25, 32, 20, 22, 13, 20, 15, 19, 
    16, 22, 14, 12, 13, 13, 6, 6, 7, 5, 9
]

# Data for Web of Science (WoS)
wos_years = [
    2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 
    2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2005
]
wos_counts = [
    17, 29, 19, 27, 23, 12, 12, 14, 16, 13, 
    11, 11, 10, 3, 7, 4, 1, 2, 1
]

scopus_manual_offsets = [
    (0.2, -2),  # 2004
    (0, 1.5),  # 2005
    (0, -1.8),  # 2006
    (0, 1),  # 2007
    (0, -1.8),  # 2008
    (0, 1),  # 2009
    (0, 1.9),  # 2010
    (0, 1),  # 2011
    (0, 1.9),  # 2012
    (0, 1),  # 2013
    (0, 1),  # 2014
    (0, 1),  # 2015
    (0, 1),  # 2016
    (0, 1),  # 2017
    (0, 1),  # 2018
    (0, 1),  # 2019
    (0, 1),  # 2020
    (0, 1),  # 2021
    (0, 1),  # 2022
    (0, 1),  # 2023
    (0, 1),  # 2024
]

wos_manual_offsets = [
    (0, -1.8),  # 2005
    (0.5, -1),  # 2007
    (0, -1.5),  # 2008
    (0, 0.5),  # 2009
    (-0.1, 0.7),  # 2010
    (0, -2),  # 2011
    (0, -1.5),  # 2012
    (0, -1.5),  # 2013
    (0, -1.8),  # 2014
    (0, 0.5),  # 2015
    (0, 0.8),  # 2016
    (0, 0.8),  # 2017
    (0, 0.8),  # 2018
    (0, 0.8),  # 2019
    (0, 0.8),  # 2020
    (0, 0.8),  # 2021
    (0, 0.8),  # 2022
    (0, 0.8),  # 2023
    (0, 0.7),  # 2024
]

# Adjusting the plot based on these manual offsets
plt.figure(figsize=(12, 6))
plt.plot(wos_years, wos_counts, marker='o', label="Web of Science", color="blue")
plt.plot(scopus_years, scopus_counts, marker='o', label="Scopus", color="orange")


# Correcting the x-axis ticks to avoid decimals
plt.xticks(range(2004, 2025), rotation=45)

# Adding data labels with manual offsets
for (x, y), (x_offset, y_offset) in zip(zip(scopus_years, scopus_counts), scopus_manual_offsets):
    plt.text(x + x_offset, y + y_offset, str(y), fontsize=14, ha='center', color="orange")

for (x, y), (x_offset, y_offset) in zip(zip(wos_years, wos_counts), wos_manual_offsets):
    plt.text(x + x_offset, y + y_offset, str(y), fontsize=14, ha='center', color="blue")

plt.title("Publications Over Time (Scopus vs Web of Science)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Record Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

