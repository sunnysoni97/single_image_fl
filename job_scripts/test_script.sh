#!/bin/bash

VAR1=abc
VAR2=True

if [ $VAR1 == "abc" -a $VAR2 == "True" ] 
then
    echo "Its $VAR1 and $VAR2"
fi
