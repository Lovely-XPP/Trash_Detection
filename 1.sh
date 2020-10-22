oldsuffix="jpeg"
newsuffix="jpg"
cur_dir=$(cd "$(dirname "$0")"; pwd)
dir="$cur_dir""/pic/"
cd $dir

for file in $(ls $dir | grep .${oldsuffix})
    do
        name=$(ls ${file} | cut -d. -f1)
        mv $file ${name}.${newsuffix}
    done
