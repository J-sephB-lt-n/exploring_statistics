
let canvasWidth = 1200;
let canvasHeight = 600;
let n_pixels_per_block_wide = 50;  // make sure that this divides neatly into canvasWidth
let n_pixels_per_block_high = 50;  // make sure that this divides neatly in canvasHeight

let n_blocks_wide = canvasWidth / n_pixels_per_block_wide;
let n_blocks_high  = canvasHeight / n_pixels_per_block_high;

let carVisionLength = 40;       // controls how far ahead of itself a car can see
let carVisionWidth = 0.5;        // angle (in radians) which controls how far apart the cars left and right vision points are from each other 

let showVisionPoints = false;    // controls whether the vision points which control ant movement are visible on the screen or not

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

// define a function which checks whether a given point is off the game board 
function check_if_pixel_off_gameboard( pnt_x, pnt_y ){
    if( pnt_x < 10 || pnt_x > (canvasWidth-10) || pnt_y < 10 || pnt_y > (canvasHeight-10) ){
        return true;
    } else{
        return false;
    }
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
        this.speed = 1;                                          // this is how many pixels the car will move per loop of the game

        this.rotationPrevStep = 0;         // store rotation (in radians) on previous step

        // define co-ordinates of car vision points:
        this.aheadVisionPoint_x = this.x + carVisionLength * Math.cos(this.carangle_rad);    
        this.aheadVisionPoint_y = this.y - carVisionLength * Math.sin(this.carangle_rad);    
        this.leftVisionPoint_x = this.x + (this.aheadVisionPoint_x-this.x)*Math.cos(-carVisionWidth) + (this.y-this.aheadVisionPoint_y)*Math.sin(-carVisionWidth);    
        this.leftVisionPoint_y = this.y + (this.aheadVisionPoint_y-this.y)*Math.cos(-carVisionWidth) + (this.aheadVisionPoint_x-this.x)*Math.sin(-carVisionWidth);    
        this.rightVisionPoint_x = this.x + (this.aheadVisionPoint_x-this.x)*Math.cos(carVisionWidth) + (this.y-this.aheadVisionPoint_y)*Math.sin(carVisionWidth);    
        this.rightVisionPoint_y = this.y + (this.aheadVisionPoint_y-this.y)*Math.cos(carVisionWidth) + (this.aheadVisionPoint_x-this.x)*Math.sin(carVisionWidth);
        
        // define starting vision point states:
        this.aheadVisionPoint_collision = false;
        this.leftVisionPoint_collision = false;
        this.rightVisionPoint_collision = false;
    }

    draw(){
        // draw circle in centre of car:
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc( this.x, this.y, 3, 0,2*Math.PI);
        ctx.stroke();
        //ctx.fill();

        // draw circle at centre of car front:
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc( this.carfront_x, this.carfront_y, 2, 0,2*Math.PI);
        ctx.stroke();
        //ctx.fill();
        
        // line from car centre to car front:
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.carfront_x, this.carfront_y)
        ctx.stroke();

        if( showVisionPoints ){
            // draw in the car's vision points:
            ctx.fillStyle = 'blue';
            ctx.beginPath();
            ctx.arc( this.aheadVisionPoint_x, this.aheadVisionPoint_y, 2, 0, 2*Math.PI);
            ctx.stroke();
            //ctx.fill();

            ctx.fillStyle = 'blue';
            ctx.beginPath();
            ctx.arc( this.leftVisionPoint_x, this.leftVisionPoint_y, 2, 0, 2*Math.PI);
            ctx.stroke();
            //ctx.fill(); 
            
            ctx.fillStyle = 'blue';
            ctx.beginPath();
            ctx.arc( this.rightVisionPoint_x, this.rightVisionPoint_y, 2, 0, 2*Math.PI);
            ctx.stroke();
            //ctx.fill();   
        }      
    }

    // function to rotate the car:
    // a changeInRadians>0 gives anti-clockwise rotation
    rotate( changeInRadians ){
        this.carangle_rad += changeInRadians;
        
        // calculate new co-ordinates of middle of car front:
        this.carfront_x = this.x + (this.carLength/2) * Math.cos(this.carangle_rad);       // x co-ordinate of middle of front bumper
        this.carfront_y = this.y - (this.carLength/2) * Math.sin(this.carangle_rad);       // y co-ordinate of middle of back bumper
        
        // calculate new co-ordinates of car vision points:
        this.aheadVisionPoint_x = this.x + carVisionLength * Math.cos(this.carangle_rad);    
        this.aheadVisionPoint_y = this.y - carVisionLength * Math.sin(this.carangle_rad);   
        this.leftVisionPoint_x = this.x + (this.aheadVisionPoint_x-this.x)*Math.cos(-carVisionWidth) + (this.y-this.aheadVisionPoint_y)*Math.sin(-carVisionWidth);    
        this.leftVisionPoint_y = this.y + (this.aheadVisionPoint_y-this.y)*Math.cos(-carVisionWidth) + (this.aheadVisionPoint_x-this.x)*Math.sin(-carVisionWidth);    
        this.rightVisionPoint_x = this.x + (this.aheadVisionPoint_x-this.x)*Math.cos(carVisionWidth) + (this.y-this.aheadVisionPoint_y)*Math.sin(carVisionWidth);    
        this.rightVisionPoint_y = this.y + (this.aheadVisionPoint_y-this.y)*Math.cos(carVisionWidth) + (this.aheadVisionPoint_x-this.x)*Math.sin(carVisionWidth);    

        // stop car angle from exceeding the (-2pi, 2pi) range:
        if( this.carangle_rad > 2*Math.PI ){
            this.carangle_rad -= 2*Math.PI;
        } else if (this.carangle_rad < -2*Math.PI) {
            this.carangle_rad += 2*Math.PI;   
        }
    } // end of Rotate() function

    // if the car is moving, then calculate it's new location:
    calc_new_car_location(){
        this.x += this.speed * Math.cos(this.carangle_rad);
        this.y -= this.speed * Math.sin(this.carangle_rad);
        this.carfront_x += this.speed * Math.cos(this.carangle_rad);
        this.carfront_y -= this.speed * Math.sin(this.carangle_rad);
    }

    check_for_vision_point_collisions(){
        this.aheadVisionPoint_collision = check_if_pixel_off_gameboard( this.aheadVisionPoint_x, this.aheadVisionPoint_y );
        this.leftVisionPoint_collision = check_if_pixel_off_gameboard( this.leftVisionPoint_x, this.leftVisionPoint_y );
        this.rightVisionPoint_collision = check_if_pixel_off_gameboard( this.rightVisionPoint_x, this.rightVisionPoint_y );
    }

    clear_vision_point_collisions(){
        this.aheadVisionPoint_collision = false;
        this.leftVisionPoint_collision = false;
        this.rightVisionPoint_collision = false;
    }

    choose_rotation(){
        if( !this.leftVisionPoint_collision && !this.aheadVisionPoint_collision && !this.rightVisionPoint_collision ){
            // if there are no vision point collisions
            this.rotationPrevStep = 0;
            //this.rotationPrevStep
        } else if(this.leftVisionPoint_collision && !this.rightVisionPoint_collision) {
            this.rotationPrevStep = -0.05;       // turn right a bit 
        } else if(!this.leftVisionPoint_collision && this.rightVisionPoint_collision){
            this.rotationPrevStep = 0.05;        // turn left a bit
        } else if(this.leftVisionPoint_collision && this.rightVisionPoint_collision){
            if( this.rotationPrevStep == 0 ){
                if( Math.random() > 0.5 ){ this.rotationPrevStep = -0.05; } else { this.rotationPrevStep = 0.05; }   
            }
        }
        //this.rotationPrevStep
    }

} // end of class [Car]

function drawGridlines(){
    // draw the gridlines:
    ctx.fillStyle = 'black';
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

function showVisionPoints_function(){
    if( showVisionPoints ){
        showVisionPoints = false;
    } else {
        showVisionPoints = true;
    }
}

function RenderGame() {
        
    // clear the game canvas:
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // iterate through the cars:
    if( cars.length > 0 ){
        for( i=0; i<cars.length; i++ ){

            cars[i].check_for_vision_point_collisions();
            cars[i].choose_rotation();
            cars[i].rotate( cars[i].rotationPrevStep );
            cars[i].calc_new_car_location();
            cars[i].clear_vision_point_collisions();
            cars[i].rotate( (Math.random()-0.5) / 4 );

            // draw the car:
            cars[i].draw();
        }
    }

    // draw the gridlines:
    drawGridlines();
    
    // keep running RenderGame:
    requestAnimationFrame(RenderGame);
}


