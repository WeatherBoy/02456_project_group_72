#!/usr/bin/env bash

source /zhome/e7/a/137819/deep_venv/bin/activate
# /zhome/e7/a/137819/deep_venv/bin/python3 -u /zhome/06/a/147115/02456_project_group_72/src/chatbot.py > /zhome/e7/a/137819/02456_project_group_72/seq_Ja/logs/$JOB_NAME/output.txt


>res_seq_Ja.txt
# hyper_params directories you want to test
for epoc in 50 #75 100 
	do

	# Loop over batch_size to test
	for batch_size in 8 12 16 24
		do

		# loop over LR
		for LR in 2e-3 1e-3 5e-3 1e-4
			do

			for max_len in 8 10 15 #100
				do
				
				#loop hidden
				for hidden in 64 128 #256 
                do

                echo $epoc $LR $max_len $hidden 
				echo $epoc $LR $max_len $hidden >> res_seq_Ja.txt
				/zhome/e7/a/137819/deep_venv/bin/python3 -u (seq_Ja.py -b $batch_size -e $epoc -LR $LR -max_l $max_len -hid $hidden) >> res_seq_Ja.txt > /zhome/e7/a/137819/02456_project_group_72/seq_Ja/logs/$JOB_NAME/output.
							


				done
            done
        done
    done
done
