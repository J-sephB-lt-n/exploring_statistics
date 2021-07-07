
let canvasWidth = 1000;
let canvasHeight = 600;
let n_pixels_per_block_wide = 50;  // make sure that this divides neatly into canvasWidth
let n_pixels_per_block_high = 50;  // make sure that this divides neatly in canvasHeight

let n_blocks_wide = canvasWidth / n_pixels_per_block_wide;
let n_blocks_high  = canvasHeight / n_pixels_per_block_high;

// define co-ordinates of grid:
let block_coords = [];
//              j
//          0 1 2 3   ...    
//      0   
//      1
//  i   2
//      3
//      .
//      .
for( i=0; i<n_blocks_high; i++ ){
    block_coords.push( [] );              // add row i
    for( j=0; j<n_blocks_wide; j++ ){     // iteratively add each column to row i
        block_coords[i].push( 
                            [   // [ [Xstart,Xend), [Ystart,Yend) ]
                                [j*n_pixels_per_block_wide,j*n_pixels_per_block_wide+n_pixels_per_block_wide],
                                [i*n_pixels_per_block_high,i*n_pixels_per_block_high+n_pixels_per_block_high]
                            ] 
                            );         
    }
}

// when html has fully loaded, then run SetupCanvas() function:
document.addEventListener('DOMContentLoaded', SetupCanvas);

function SetupCanvas(){
    canvas = document.getElementById("game_canvas");
    ctx = canvas.getContext("2d");
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    // fill the canvas with black:
    ctx.fillStyle = "green";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    RenderGame();
}

// define a car object:
// (just a block for now)
class Car {
    // define car attributes:
    constructor(){
        this.x = Math.round(Math.random()*canvasWidth);            // x co-ordinate of middle of car
        this.y = Math.round(Math.random()*canvasHeight);           // y co-ordinate of middle of car
        this.carLength = 20                                       // length of car from back to front
        this.carangle_rad = -Math.PI * 1.3;                          // angle that front of car is facing (in radians)  
        this.carfront_x = this.x + (this.carLength/2) * Math.cos(this.carangle_rad)       // x co-ordinate of middle of front bumper
        this.carfront_y = this.y - (this.carLength/2) * Math.sin(this.carangle_rad)       // y co-ordinate of middle of back bumper
        this.speed = 2;                                          // this is how many pixels the car will move per loop of the game
    }

    draw(){
        // draw circle in centre of car:
        ctx.beginPath();
        ctx.arc( this.x, this.y, 3, 0,2*Math.PI);
        ctx.stroke();

        // draw circle at centre of car front:
        ctx.beginPath();
        ctx.arc( this.carfront_x, this.carfront_y, 2, 0,2*Math.PI);
        ctx.stroke();
        
        // line from car centre to car front:
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.carfront_x, this.carfront_y)
        ctx.stroke();

        // draw top left corner:
        //ctx.beginPath();
        //ctx.arc( this.carborder_topleft_x, this.carborder_topleft_y, 2, 0,2*Math.PI);
        // ctx.stroke();
        // draw bottom right corner:
        //ctx.beginPath();
        //ctx.arc( this.carborder_botright_x, this.carborder_botright_y, 2, 0,2*Math.PI);
        //ctx.stroke();
    }

    // function to rotate the car:
    // a changeInRadians>0 gives positive rotation
    rotate( changeInRadians ){
        //let temp_carfront_x = this.carfront_x;
        //let temp_carfront_y = this.carfront_y;
        //this.carfront_x = Math.cos(theta) * (temp_carfront_x-this.x) - Math.sin(theta) * (temp_carfront_y-this.y) + this.x;
        //this.carfront_y = Math.sin(theta) * (temp_carfront_x-this.x) + Math.cos(theta) * (temp_carfront_y-this.y) + this.y;
        this.carangle_rad += changeInRadians;
        this.carfront_x = this.x + (this.carLength/2) * Math.cos(this.carangle_rad)       // x co-ordinate of middle of front bumper
        this.carfront_y = this.y - (this.carLength/2) * Math.sin(this.carangle_rad)       // y co-ordinate of middle of back bumper

        // stop car angle from exceeding the (-2pi, 2pi) range:
        if( this.carangle_rad > 2*Math.PI ){
            this.carangle_rad -= 2*Math.PI
        } else if (this.carangle_rad < -2*Math.PI) {
            this.carangle_rad += 2*Math.PI    
        }
    } // end of Rotate() function

    // if the car is moving, then calculate it's new location:
    calc_new_car_location(){
        this.x += this.speed * Math.cos(this.carangle_rad);
        this.y -= this.speed * Math.sin(this.carangle_rad);
        this.carfront_x += this.speed * Math.cos(this.carangle_rad);
        this.carfront_y -= this.speed * Math.sin(this.carangle_rad);
    }
    //calc_location(){
        //let runif01 = Math.random()
        //if( runif01 < 0.5 ){
        //rotate_car( 0.1 );    
        //} else{
            // this.rotate_car( -0.1 )    
        //}
    //}

} // end of class [Car]

function drawGridlines(){
    // draw the gridlines:
    for( j=0; j<=n_blocks_wide; j++ ){   // draw in vertical lines 
        ctx.beginPath();
        ctx.moveTo(j*n_pixels_per_block_wide, 0);       // line start co-ords (x,y)
        ctx.lineTo(j*n_pixels_per_block_wide, canvasHeight);   // line end co-ords (x,y)
        ctx.stroke();
    }
    for( i=0; i<=n_blocks_high; i++ ){   // draw in horizontal lines 
        ctx.beginPath();
        ctx.moveTo(0, i*n_pixels_per_block_high);       // line start co-ords (x,y)
        ctx.lineTo(canvasWidth, i*n_pixels_per_block_high);       // line end co-ords (x,y)
        ctx.stroke();
    }
    // draw text numbering the cells:
    for( i=0; i<n_blocks_high; i++ ){
        block_coords.push( [] );              // add row i
        for( j=0; j<n_blocks_wide; j++ ){     // iteratively add each column to row i
            //ctx.fillStyle = 'green';
            //ctx.font = '60px san-serif';
            ctx.textAlign = 'center'
            ctx.fillText(   
                            i.toString()+j.toString(),     // text to write
                            ( block_coords[i][j][0][1] + block_coords[i][j][0][0] ) /2,       // x co-ordinate 
                            ( block_coords[i][j][1][1] + block_coords[i][j][1][0] ) /2,       // y co-ordinate 
                        );
        }
    }    

}

let cars = [];                  // list to store all of the cars
// car1 = new Car();

function addCar(){
    cars.push( new Car() );
}

function RenderGame() {
        
    // clear the game canvas:
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // iterate through the cars:
    if( cars.length > 0 ){
        for( i=0; i<cars.length; i++ ){
            cars[i].calc_new_car_location();
            cars[i].rotate( ( Math.random()-0.5 ) );

            // draw the car:
            cars[i].draw();
        }
    }
    // car1.rotate(0.1);
    //car1.calc_new_car_location()
    //car1.rotate( ( Math.random()-0.5 ) )

    // draw the car:
    //car1.draw();

    // draw the gridlines:
    drawGridlines();
    
    // keep running RenderGame:
    requestAnimationFrame(RenderGame);
}


