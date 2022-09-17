library(tidyverse)

setwd("~/exploring_statistics/exploring_statistics/")
setwd("C:/Users/jbolton/Documents/blogs/joe_website/")

# process for adding a post:
# 
# 1. copy rmd into main directory and [post_backup_without_navbar] folder 
# 2. add html name to index.Rmd
# 3. add html name to navbar dropdown using _site.yml
# 4. render site
# 5. add google analytics tag to main sites
# 6. replace all posts (htmls) in main folder with their non-navbar versions (from [post_backup_without_navbar] folder)
# 7. add google analytics tag & navbar to all using the code in this script

rmarkdown::render_site()



####################################################
## code to add google analytics tag to main pages ##
####################################################

google_analytics_global_site_tag <- 
"

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src=\"https://www.googletagmanager.com/gtag/js?id=UA-163824605-1\"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-163824605-1');
</script>
"

for( file_i in c("index.html", "about.html", "future_content.html") ){
  
  cat("processing [[", file_i, "]]...\n" )  
  
  readin_html_text <- readr::read_lines(file_i)
  
  line_to_insert_google_analytics_global_site_tag_code <-
    min( grep('<head>', readin_html_text) ) +1
  
  complete_html <-
    paste0(
      # html before google analytics global site tag code:
      paste( readin_html_text[1:(line_to_insert_google_analytics_global_site_tag_code-1)],
             collapse="\n"
      )
      ,
      # google analytics global site tag code:
      google_analytics_global_site_tag
      ,
      # html after google analytics global site tag code:
      paste( readin_html_text[line_to_insert_google_analytics_global_site_tag_code:length(readin_html_text)],
             collapse="\n"
      )
    )
  
  readr::write_lines(    x = complete_html,
                         file = file_i
  )
}

###############################################################
## code to add navbar and google analytics tag to blog posts ##
###############################################################

# <<<<<<<<< WARNING >>>>>>>>>>
# <<<<<<<<< THE BACKUPS WITHOUT NAVBAR CODE ARE NOT OVERWRITTEN! 
# <<<<<<<<< MAKE SURE THAT THESE ARE UP TO DATE BEFORE RUNNING THIS PROCESS!

# get the navbar code from index.html and paste it here:

navbar_padding_code <- 
'

<style type="text/css">
/* padding for bootstrap navbar */
body {
  padding-top: 60px;
  padding-bottom: 40px;
}
/* offset scroll position for anchor links (for fixed navbar)  */
.section h1 {
  padding-top: 65px;
  margin-top: -65px;
}
.section h2 {
  padding-top: 65px;
  margin-top: -65px;
}
.section h3 {
  padding-top: 65px;
  margin-top: -65px;
}
.section h4 {
  padding-top: 65px;
  margin-top: -65px;
}
.section h5 {
  padding-top: 65px;
  margin-top: -65px;
}
.section h6 {
  padding-top: 65px;
  margin-top: -65px;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu>.dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
  border-radius: 0 6px 6px 6px;
}
.dropdown-submenu:hover>.dropdown-menu {
  display: block;
}
.dropdown-submenu>a:after {
  display: block;
  content: " ";
  float: right;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
  border-width: 5px 0 5px 5px;
  border-left-color: #cccccc;
  margin-top: 5px;
  margin-right: -10px;
}
.dropdown-submenu:hover>a:after {
  border-left-color: #ffffff;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left>.dropdown-menu {
  left: -100%;
  margin-left: 10px;
  border-radius: 6px 0 6px 6px;
}
</style>

'

navbar_creation_code <-      # this code can be pasted in from index.html after rendering the site
'  
  
  
  
<div class="navbar navbar-inverse  navbar-fixed-top" role="navigation">
  <div class="container">
    <div class="navbar-header">
      <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar">
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
      </button>
      <a class="navbar-brand" href="index.html">Joseph Bolton</a>
    </div>
    <div id="navbar" class="navbar-collapse collapse">
      <ul class="nav navbar-nav">
        <li>
  <a href="index.html"> Home</a>
</li>
<li>
  <a href="about.html"> About Me</a>
</li>
<li>
  <a href="future_content.html"> Future Content</a>
</li>
<li class="dropdown">
  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false">
     Posts
     
    <span class="caret"></span>
  </a>
  <ul class="dropdown-menu" role="menu">
    <li class="dropdown-header">Posts</li>
    <li>
      <a href="document_similarity_recommenders.html">A Selective Survey of Word &amp; Document Embedding Methods</a>
    </li>
    <li>
      <a href="dcn_v2_cross_layer_investigation.html">Dissecting the DCN-V2 Cross Layer</a>
    </li>
    <li>
      <a href="cuped_cupac_and_other_variance_reduction_techniques.html">CUPED, CUPAC, and Other Ways to Reduce Variance in an Experiment</a>
    </li>
    <li>
      <a href="recommenders_part1_vectors.html">Recommenders Part I: Representing Customers &amp; Items as Vectors</a>
    </li>
    <li>
      <a href="nice_R_visualisations.html">Beautiful Lesser-Known Visualisations in R</a>
    </li>
    <li>
      <a href="gbm_from_scratch_8020.html">Building a GBM from Scratch</a>
    </li>
    <li>
      <a href="good_books_and_papers.html">My Favourite Books, Papers and other Resources</a>
    </li>
    <li>
      <a href="neural_net_from_scratch.html">Deep-Learning from Scratch</a>
    </li>
    <li>
      <a href="visual_intro_to_splines.html">Regression Splines</a>
    </li>
    <li>
      <a href="PAMS_and_SILHOUETTE_by_hand.html">Clustering: PAM k-Medoids, CLARA &amp; Silhouette Values</a>
    </li>
    <li>
      <a href="intuition_for_CRVTW_uplift_model.html">CRVTW Revenue Uplift Model</a>
    </li>
    <li>
      <a href="data_tree_checkout.html">Data.trees</a>
    </li>
    <li>
      <a href="constrOptim.html">Constrained Optimisation in R</a>
    </li>
    <li>
      <a href="OLS_theory.html">Least Squares Linear Model Theory</a>
    </li>
  </ul>
</li>
      </ul>
      <ul class="nav navbar-nav navbar-right">
        <li>
  <a href="mailto:joseph.jazz.bolton@gmail.com">
    <span class="fa fa-envelope"></span>
     
  </a>
</li>
<li>
  <a href="https://www.linkedin.com/in/joseph-bolton-a02653171/">
    <span class="fa fa-linkedin"></span>
     
  </a>
</li>
      </ul>
    </div><!--/.nav-collapse -->
  </div><!--/.container -->
</div><!--/.navbar -->
  
'
 
# specify name(s) of posts to insert navbar into (as a character vector): 
post_names_to_insert_navbar <- 
   c( 
      "visual_intro_to_splines.html",
      "constrOptim.html",
      "data_tree_checkout.html",
      "gbm_from_scratch_8020.html",
      "good_books_and_papers.html",
      "intuition_for_CRVTW_uplift_model.html",
      "neural_net_from_scratch.html",
      "nice_R_visualisations.html",
      "OLS_theory.html",
      "PAMS_and_SILHOUETTE_by_hand.html",
      "recommenders_part1_vectors.html",
      "cuped_cupac_and_other_variance_reduction_techniques.html",
      "dcn_v2_cross_layer_investigation.html",
      "document_similarity_recommenders.html"
    )

for( post_i in post_names_to_insert_navbar ){
  
  cat("processing [[", post_i, "]]...\n" )  
  
  # back up post before inserting navbar code, if this has not already been done:
  if( !file.exists( paste0("./post_backup_without_navbar/", post_i) ) ){
    file.copy(      from = post_i,
                      to = paste0("./post_backup_without_navbar/", post_i)
             )
  }
  
  readin_html_text <- readr::read_lines(post_i)
  
  line_to_insert_google_analytics_global_site_tag_code <-
    min( grep('<head>', readin_html_text) ) +1
  
  line_to_put_navbar_padding_code <-
    min( grep('<!-- tabsets -->', readin_html_text) ) -2

  line_to_put_navbar_creation_code <-
    grep('<div class="container-fluid main-container">', readin_html_text) + 1

  complete_html <-
    paste0(
      # html before google analytics tag code:
      paste( readin_html_text[1:(line_to_insert_google_analytics_global_site_tag_code-1)],
             collapse="\n"
      ),
      
      # google analytics tag code:
      google_analytics_global_site_tag,
      
      # html between google analytics tag and code and before navbar code:
      paste( readin_html_text[line_to_insert_google_analytics_global_site_tag_code:(line_to_put_navbar_padding_code-1)],
             collapse="\n"
           )
      ,
      # navbar padding code:
      navbar_padding_code
      ,
      # html between navbar padding code and navbar creation code:
      paste( readin_html_text[line_to_put_navbar_padding_code:(line_to_put_navbar_creation_code-1)],
             collapse="\n"
           )
      ,
      # navbar creation code:
      navbar_creation_code
      ,
      paste( readin_html_text[line_to_put_navbar_padding_code:length(readin_html_text)],
             collapse="\n"
           )
    )

  readr::write_lines(    x = complete_html,
                      file = post_i
                    )
}

