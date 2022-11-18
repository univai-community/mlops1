# mlops1

```
pip install dvc
```

If you are forking/cloning this repository, make sure you

```
rm -rf .dvc, *.dvc ~/warehouse
```

Now the dvc commands are:

```
dvc init
dvc remote add warehouse ~/warehouse
dvc remote default warehouse
dvc add data
dvc add artifacts
dvc status
```

This will create some `.dvc` files for you
```
cat artifacts.dvc
cat data.dvc
git add .; git commit -m "added dvc files" -a; git push
dvc status
dvc push
```

Now copy the csv files from the playdata folder into data. These are not tracked by git. Then:

```
dvc add data
dvc status
cat data.dvc # this has changed
git add .;git commit -m "added csv files" -a; git push
dvc push
```

Now the files are in the warehouse. Use `dvc pull` ifyou are working with someone, or you have a fresh clone somewhere else.

Later, after you run the second notebook and have artifacts:

```
dvc add artifacts
dvc status
cat artifacts.dvc
git add .;git commit -m "added artifact files" -a; git push
dvc push
```

The git and dvc together make sure that data is tracked with code.