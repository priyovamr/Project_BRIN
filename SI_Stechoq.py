import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
divisi = ['IT', 'Sales', 'Marketing', 'Accounting']
jml_karyawan = [50, 25, 30, 20]
rata_gaji = [8, 6, 7, 5]

# Create a DataFrame
data = {'Divisi': divisi, 'Jumlah_karyawan': jml_karyawan, 'Rata-rata gaji (dalam juta rupiah)': rata_gaji}
df = pd.DataFrame(data)

# Plot the number of employees per division
plt.figure(figsize=(10, 5))
plt.bar(df['Divisi'], df['Jumlah_karyawan'], color='blue')
plt.title('Jumlah karyawan per divisi')
plt.xlabel('Divisi')
plt.ylabel('Jumlah karyawan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the average salary per division
plt.figure(figsize=(10, 5))
plt.bar(df['Divisi'], df['Rata-rata gaji (dalam juta rupiah)'], color='darkgreen')
plt.title('Rata-rata gaji per divisi')
plt.xlabel('Divisi')
plt.ylabel('Rata-rata gaji (dalam juta rupiah)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()