# From md to word

pandoc -f docx -t markdown -o /Users/Thomas/Downloads/R_collaborative.md /Users/Thomas/Downloads/R_collaborative.docx

## Grammarly



## Change the pictures

cd /Users/Thomas/Downloads/

find . -type f -name "R_collaborative.md" -exec sed -i'' -e 's/media/\/Users\/Thomas\/Dropbox\/Learning\/GitHub\/project\/thomaspernet\/static\/project\/collaboration/g' {} +



