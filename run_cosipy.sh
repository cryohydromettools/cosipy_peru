
cd optimisation/

python3 free_parameter_line.py

cd ../

cp optimisation/constants.py ../cosipy_peru/

cd utilities/aws2cosipy

python3 aws2cosipy.py -c ../../data/input/Peru/data_aws_peru.csv -o ../../data/input/Peru/Peru_stake.nc -s ../../data/static/Peru_static_stake.nc -b 20160901 -e 20170831

cd ../../

# config file

line=0

line_string="output_netcdf = 'Peru_C$line'+'_'+time_start_str+'-'+time_end_str+'.nc'"
echo $line_string
line_par=31

filename_out2="config.py"

awk 'NR==n{$0=c}1' n="$line_par" c="$line_string" $filename_out2 > tmp && mv tmp $filename_out2

python3 COSIPY.py

cd postprocessing/post_cosipy/

python3 SEB_mean.py
python3 validation_model_stake.py

cd ../../

#python3 postprocessing/post_cosipy/SEB_mean.py



