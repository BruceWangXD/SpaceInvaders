
package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;


public class Projectile extends GameObject {
    //declare image
    private PImage img;
    private int damage;
    //constructor
    public Projectile(PImage img, int x, int y, int width, int height,int health,int velocity,int damage) {
        super(x, y, width, height,health,velocity);
        this.img = img;
        this.damage = damage;
    }
    //tick the projectile
    public void tick() {
        this.y -= velocity;
    }
    //draw the projectile
    public void draw(PApplet app) {
        app.image(img, x, y, width, height);
        tick();
    }
    //return the damage of the projectile
    public int getDamage(){
        return damage;
    }
    public boolean intersect(GameObject r2){
        if ( this.getX() < r2.getX() + r2.getWidth() && this.getX() + this.getWidth() > r2.getX() && this.getY() < r2.getY() + r2.getHeight() && this.getHeight() + this.getY() > r2.getY()){
            return true;
        }
        return false;
    }
    public boolean isFriendly(){
        if(velocity > 0){
            return true;
        }
        return false;
    }
}
