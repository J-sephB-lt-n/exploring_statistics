

category_vbl <- iris$Species
continuous_vbl <- iris$Sepal.Width

geom_flat_violin <- 
  function(mapping = NULL, data = NULL, stat = "ydensity",
           position = "dodge", trim = TRUE, scale = "area",
           show.legend = NA, inherit.aes = TRUE, ...) {
    layer(
      data = data,
      mapping = mapping,
      stat = stat,
      geom = GeomFlatViolin,
      position = position,
      show.legend = show.legend,
      inherit.aes = inherit.aes,
      params = list(
        trim = trim,
        scale = scale,
        ...
      )
    )
  }

raincloud_plot_function <- 
  function(      category_vbl,
               continuous_vbl
          ){
    
    cat("##### CREDIT TO #####\n    concept: Paula Andrea Martinez, Dr. Micah Allen, David Robinson \n additional: David Zhao \n     source: https://orchid00.github.io/tidy_raincloudplot \n             https://towardsdatascience.com/the-ultimate-eda-visualization-in-r-e6aff6afe5c1 \n#####################")
    
    n_palette_colours <- length( unique(category_vbl) )+2
    
    getPalette <- colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_palette_colours)
    
    source("https://gist.githubusercontent.com/benmarwick/2a1bb0133ff568cbe28d/raw/fb53bd97121f7f9ce947837ef1a4c65a73bffb3f/geom_flat_violin.R")
    
    lwr0.25 <- function(x) quantile(x, probs=0.25)
    upr0.75 <- function(x) quantile(x, probs=0.75)
    
    data_for_plot <-  tibble( x_var = category_vbl,
                              y_var = continuous_vbl 
                            )
    
    sumld <- data_for_plot %>% 
                group_by(x_var) %>% 
                summarise_all( tibble::lst(mean, median, lower0.25=lwr0.25, upper0.75=upr0.75) )
    
    print( sumld )
    
    create_ggplot <- 
      ggplot( data = data_for_plot, 
             aes(    x = x_var, 
                     y = y_var, 
                  fill = x_var 
                )
             ) +
      geom_flat_violin( position = position_nudge(x = .2, y = 0), trim = TRUE, alpha = .8, scale = "width") +
      geom_point( aes(    y = y_var, 
                      color = x_var
                     ), 
                  position = position_jitter(width = .15),
                      size = 0.5, 
                     alpha = 0.8
                ) +
      geom_boxplot(         width = 0.1, 
                    outlier.shape = NA,
                            alpha = 0.5
                  ) +
      geom_point( data = sumld, aes(x = factor(country), y = mean), 
                 position = position_nudge(x = 0.3), size = 2.5) +
      geom_errorbar(data = sumld, aes(ymin = lower, ymax = upper, y = mean), 
                    position = position_nudge(x = 0.3), width = 0)+
      expand_limits(x = 5.25) +
      guides(fill = FALSE) +
      guides(color = FALSE) +
      scale_color_manual(values = getPalette) +
      scale_fill_manual(values = getPalette) +
      #coord_flip() + # flip or not
      theme_bw() +
      raincloud_theme +
      theme(axis.title = element_text(size = 42),
            axis.text=element_text(size=42))
    
  }