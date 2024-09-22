#!/bin/bash

os_type=$(uname)

if [[ "$os_type" == "Darwin" ]]; then
  brew install nim
elif [[ "$os_type" == "Linux" ]]; then
  apt-get install nim
else
  echo "This is not a Mac or Linux system."
  exit
fi

nimble install nimpy
