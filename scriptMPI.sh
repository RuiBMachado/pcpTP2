cd pcp/TP2
module load gnu/4.9.0
module load gnu/openmpi_mx/1.8.4
mpicc -std=c99 -o heatSimulatorMPI heatSimulatorMPI.c


for X in {1..5}
do
echo "REPETICAO $X" >> ReportMPI.txt
	for asd in 5
	do
		echo "Processo $asd" >> ReportMPI.txt
		mpirun -np $asd -mca btl self,sm,tcp heatSimulatorMPI 0.001  >> ReportMPI.txt
		echo "" >> ReportMPI.txt
	done
done