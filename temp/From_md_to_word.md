# From md to word

pandoc -f docx -t markdown -o /Users/Thomas/Downloads/R_collaborative.md /Users/Thomas/Downloads/R_collaborative.docx

## Grammarly



## Change the pictures

cd /Users/Thomas/Downloads/

find . -type f -name "R_collaborative.md" -exec sed -i'' -e 's/media/\/static\/project\/collaboration/g' {} +



Need to replace with < scr



remove the {}

```
find . -type f -name "25_Google_Cloud_Platform.md" -exec perl -i -0777 -pe 's/{.*?}//sg' {} +
```