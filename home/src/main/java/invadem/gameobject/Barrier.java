
package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;

import java.util.ArrayList;
import java.util.Iterator;


public class Barrier{
    //coordinate of the barrier
    private int x;
    private int y;
    //list of the barrierComponent
    private ArrayList<BarrierComponent> cs = new ArrayList<>();
    //constructor for ArmouredInvader
    public Barrier(int x, int y,BarrierComponent left, BarrierComponent right, BarrierComponent solid, BarrierComponent top) {
        this.x = x;
        this.y = y;
        cs.add(solid.copy(x,y+16));
        cs.add(solid.copy(x,y+8));
        cs.add(left.copy(x,y));
        cs.add(top.copy(x+8,y));
        cs.add(right.copy(x+16,y));
        cs.add(solid.copy(x+16,y+8));
        cs.add(solid.copy(x+16,y+16));
    }
    //draw the whole
    public void draw(PApplet app) {
        for(BarrierComponent i : cs){
            i.draw(app);
        }
    }

    public ArrayList<BarrierComponent> getCs() {
        return cs;
    }
}
