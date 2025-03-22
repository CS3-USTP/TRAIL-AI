#!/bin/bash
ollama serve &
sleep 2
ollama pull gemma2:2b-instruct-q4_0
wait