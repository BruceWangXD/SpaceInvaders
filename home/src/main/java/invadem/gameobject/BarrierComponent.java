package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;


public class BarrierComponent extends GameObject {
    //declare different kinds of images
    private PImage imgOne;
    private PImage imgTwo;
    private PImage imgThree;
    private PImage imgFour;
    //constructor
    public BarrierComponent(PImage imgOne, PImage imgTwo, PImage imgThree, PImage imgFour, int x, int y, int width, int height,int health,int velocity) {
        super(x, y, width, height,health,velocity);
        this.imgOne = imgOne;
        this.imgTwo = imgTwo;
        this.imgThree = imgThree;
        this.imgFour = imgFour;
    }
    //draw the barrier component
    public void draw(PApplet app) {
        if(health == 3){
            app.image(imgOne, x, y, width, height);
        }else if(health == 2){
            app.image(imgTwo, x, y, width, height);
        }else if(health == 1){
            app.image(imgThree, x, y, width, height);
        }else{
            app.image(imgFour, x, y, width, height);
        }
    }
    //copy a barrier
    public BarrierComponent copy(int newX, int newY){
        return new BarrierComponent(this.imgOne, this.imgTwo, this.imgThree,this.imgFour, newX, newY, this.width, this.height, this.health,this.velocity);
    }
    //when the barrier component be attacked
    public void attacked(Projectile p){
        this.health -= p.getDamage();
    }
    //set the health of the barrier
    public void setHealth(int health) {
        this.health = health;
    }
}