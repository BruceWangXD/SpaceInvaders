package invadem;

import invadem.gameobject.Barrier;
import invadem.gameobject.BarrierComponent;
import invadem.gameobject.Invader;
import org.junit.Test;
import processing.core.PApplet;
import processing.core.PImage;

import static org.junit.Assert.assertNotNull;

public class BarrierTest {
    @Test
    public void testBarrierConstruction() {
        BarrierComponent solid = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent top = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent right = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent left = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        Barrier barrier = new Barrier(200,430,left, right, solid, top);
        assertNotNull(barrier);
    }
    @Test
    public void testBarrierGetCs() {
        BarrierComponent solid = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent top = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent right = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        BarrierComponent left = new BarrierComponent(null,null,null,null,0,0,8,8,3,0);
        Barrier barrier = new Barrier(200,430,left, right, solid, top);
        assertNotNull(barrier.getCs());
    }
}
