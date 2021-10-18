para1_arr=("0.0001" "1e-05" "1e-06" "1e-07" "1e-08")
para2_arr=(5)


#data_arr=("youtube" "orkut-links" "twitter" "friendster")
data_arr=("toy")

echo "=== Experiments in local clustering with HKPR ==="
rm HKPR
make
for data_name in "${data_arr[@]}"
do	
	for((j=0;j<${#para2_arr[@]};j++))
	do
		echo "./HKPR -d ./ -f ${data_name} -algo powermethod -qn 10 -t ${para2_arr[$j]}" |bash;
		for((i=0;i<${#para1_arr[@]};i++))
		do
			echo "./HKPR -d ./ -f ${data_name} -algo AGP -e ${para1_arr[$i]} -qn 10 -t ${para2_arr[$j]}" |bash;
		done
	done
done



